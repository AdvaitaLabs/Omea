"""Post-generation Vina evaluation of both frozen CNN-only ablations.

Never writes into the generating run or feeds scores back to generation/selection.
Run with the standard Python environment; GNINA is invoked as a separate process.
"""
import concurrent.futures as futures
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/cnn_clean_ablations_scale30_v1'
SOURCE = ROOT / 'results/direct_guidance_selected_x0p02_h0p6_scale30_v1'
OUT = ROOT / 'results/cnn_clean_dual_evaluation_v1'
GNINA = Path('/root/autodl-tmp/rgroup-judge-benchmark/bin/gnina-1.1')
ARMS = ['zero', 'x0p02_h0p6', 'cnn_element_late_taper', 'cnn_local_geometry_guard']
WORKERS = 16


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def status(stage, **kwargs):
    value = dict(stage=stage, updated_utc=now(), pid=os.getpid(), **kwargs)
    write(OUT / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def key(row):
    return row['arm'], row['case_id'], row['attempt']


def good(row):
    s, a = row.get('raw_score', {}), row['raw_audit']
    checks = s.get('posebusters_checks', {})
    required = read(ROOT / 'results/selection_rules_v3/geometry_schema.json')['required_checks']
    return bool(a.get('strict_success') and a.get('new_design') and s.get('status') == 'scored'
                and s.get('posebusters_complete') and s.get('posebusters_pass')
                and all(checks.get(check) is True for check in required))


def score(job, mode='none', validation=False):
    directory = OUT / ('validation_full' if validation else 'scores') / job['arm'] / job['case_id'] / f"{job['attempt']:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'score.json'
    command = [str(GNINA), '-r', job['protein_path'], '-l', job['input_path'],
               '--scoring', 'vina', '--seed', '20260930', '--cpu', '2', '--no_gpu',
               '--cnn_scoring', mode, '--score_only']
    assert sha(job['input_path']) == job['input_sha256'], job
    assert sha(job['protein_path']) == job['protein_sha256'], job
    if target.exists():
        saved = read(target)
        assert saved['command'] == command and saved['input_sha256'] == job['input_sha256']
        assert saved['protein_sha256'] == job['protein_sha256']
        if saved['status'] == 'scored':
            return saved
    start = time.monotonic()
    result = dict(job, command=command, created_utc=now(), status='failed')
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=120)
        (directory / 'gnina.log').write_text(proc.stdout)
        result['log_sha256'] = sha(directory / 'gnina.log')
        assert proc.returncode == 0, f'GNINA exit {proc.returncode}'
        match = re.search(r'^Affinity:\s*([-+0-9.eE]+)', proc.stdout, re.M)
        assert match is not None, 'Missing Vina Affinity output'
        result['vina_kcal_mol'] = float(match.group(1))
        assert math.isfinite(result['vina_kcal_mol'])
        if mode == 'rescore':
            for label, name in [('CNNaffinity', 'cnn_affinity_pK'), ('CNNscore', 'cnn_pose_score')]:
                match = re.search(rf'^{label}:\s*([-+0-9.eE]+)', proc.stdout, re.M)
                assert match is not None
                result[name] = float(match.group(1))
        result['status'] = 'scored'
    except Exception as exc:
        result['error'] = repr(exc)
    result['wall_seconds'] = time.monotonic() - start
    write(target, result)
    return result


def validate(new_jobs, source_rows, references, tasks):
    # Chosen before scores are observed: first scored attempt for each arm/case.
    first = {}
    for job in new_jobs:
        first.setdefault((job['arm'], job['case_id']), job)
    control_jobs = []
    for case, task in tasks.items():
        old = next((r for r in source_rows if r['arm'] == 'x0p02_h0p6'
                   and r['case_id'] == case and r.get('raw_score', {}).get('status') == 'scored'), None)
        candidates = [('reference_validation', 0, references[case])]
        if old is not None:
            candidates.append(('control_validation', old['attempt'], old['raw_score']))
        for arm, attempt, s in candidates:
            command = s['command']
            control_jobs.append(dict(arm=arm, case_id=case, attempt=attempt,
                input_path=command[command.index('-l') + 1], input_sha256=s['input_sha256'],
                protein_path=task['protein_path'], protein_sha256=sha(task['protein_path']),
                expected_vina=s['vina_kcal_mol']))
    def check_new(job):
        fast, full = score(job), score(job, 'rescore', True)
        assert fast['status'] == full['status'] == 'scored', (fast, full)
        assert fast['vina_kcal_mol'] == full['vina_kcal_mol'], (fast, full)
        return dict(arm=job['arm'], case_id=job['case_id'], attempt=job['attempt'],
                    empirical_only=fast['vina_kcal_mol'], original_full_command=full['vina_kcal_mol'], passed=True)
    def check_control(job):
        result = score(job)
        assert result['status'] == 'scored', result
        assert result['vina_kcal_mol'] == job['expected_vina'], result
        return dict(arm=job['arm'], case_id=job['case_id'], attempt=job['attempt'],
                    empirical_only=result['vina_kcal_mol'], frozen_vina=job['expected_vina'], passed=True)
    checked = []
    with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = [pool.submit(check_new, j) for j in first.values()]
        pending += [pool.submit(check_control, j) for j in control_jobs]
        for f in futures.as_completed(pending):
            checked.append(f.result())
            if len(checked) % 20 == 0:
                status('validating', completed=len(checked), total=len(pending))
    assert len(checked) == len(first) + len(control_jobs)
    result = dict(passed=True, exact_vina_matches=len(checked),
        original_command_comparisons=len(first),
        frozen_control_comparisons=sum(j['arm'] == 'control_validation' for j in control_jobs),
        frozen_reference_comparisons=sum(j['arm'] == 'reference_validation' for j in control_jobs),
        note='Cases with no strict-task passing output have no generated molecule to cross-check; all their attempts remain in the denominator.', rows=checked)
    write(OUT / 'validation.json', result)
    return result


def summarize(records, references, tasks, original_accounting):
    # Cache the original validity decisions before repeated scope aggregation.
    for row in records:
        row['evaluation_valid_new'] = good(row)
    def success(row, metric):
        if not row['evaluation_valid_new']:
            return False
        s, ref = row['raw_score'], references[row['case_id']]
        cnn = s['cnn_affinity_pK'] > ref['cnn_affinity_pK']
        vina = s.get('vina_kcal_mol') is not None and s['vina_kcal_mol'] < ref['vina_kcal_mol']
        return {'cnn': cnn, 'vina': vina, 'dual': cnn and vina}[metric]
    def scope(case_ids):
        wanted = set(case_ids)
        expected = {(case, i) for case in wanted for i in range(100)}
        by = {arm: {(r['case_id'], r['attempt']): r for r in records
                    if r['arm'] == arm and r['case_id'] in wanted} for arm in ARMS}
        assert all(set(rows) == expected for rows in by.values())
        arms = {}
        for arm, mapping in by.items():
            rows = list(mapping.values())
            count = len(rows)
            valid = [r for r in rows if r['evaluation_valid_new']]
            m = dict(attempts=count, valid_new=len(valid), valid_new_rate=len(valid)/count,
                     scored=sum(r.get('raw_score', {}).get('status') == 'scored' for r in rows),
                     vina_scored=sum('vina_kcal_mol' in r.get('raw_score', {}) for r in rows))
            for metric in ('cnn', 'vina', 'dual'):
                m[metric + '_success'] = sum(success(r, metric) for r in rows)
                m[metric + '_success_rate'] = m[metric + '_success']/count
            m['valid_cnn_only'] = m['cnn_success'] - m['dual_success']
            m['valid_vina_only'] = m['vina_success'] - m['dual_success']
            m['valid_neither'] = len(valid) - m['cnn_success'] - m['vina_success'] + m['dual_success']
            m['remaining_to_18pct'] = max(0, math.ceil(count * .18) - m['dual_success'])
            m['remaining_to_strictly_above_18pct'] = max(0, math.floor(count * .18) + 1 - m['dual_success'])
            arms[arm] = m
        base = by['x0p02_h0p6']
        paired = {}
        for arm in ARMS[2:]:
            gained = [k for k in sorted(expected) if success(by[arm][k], 'dual') and not success(base[k], 'dual')]
            lost = [k for k in sorted(expected) if success(base[k], 'dual') and not success(by[arm][k], 'dual')]
            shared = [k for k in sorted(expected) if by[arm][k]['evaluation_valid_new'] and base[k]['evaluation_valid_new']]
            differences = [by[arm][k]['raw_score']['vina_kcal_mol'] - base[k]['raw_score']['vina_kcal_mol'] for k in shared]
            paired[arm] = dict(dual_gained=len(gained), dual_lost=len(lost), dual_net=len(gained)-len(lost),
                delta_percentage_points=100*(len(gained)-len(lost))/len(expected),
                gained_attempts=gained, lost_attempts=lost, both_valid_new=len(shared),
                vina_delta_both_valid_mean=statistics.mean(differences) if differences else None,
                vina_delta_both_valid_median=statistics.median(differences) if differences else None)
            assert paired[arm]['dual_net'] == arms[arm]['dual_success'] - arms['x0p02_h0p6']['dual_success']
        return dict(cases=len(wanted), attempts_per_arm=len(expected), arms=arms, paired_vs_cnn_baseline=paired)
    ids = list(tasks)
    previous = set(read(RUN / 'manifest.json')['previous_selection_case_ids'])
    scopes = {name: scope(cases) for name, cases in [
        ('all_30', ids), ('previous_selection_10', [c for c in ids if c in previous]),
        ('previous_additional_20', [c for c in ids if c not in previous])]}
    per_case = {case: scope([case]) for case in ids}
    per_target = {group: scope([c for c, t in tasks.items() if t['target_group'] == group])
                  for group in sorted({t['target_group'] for t in tasks.values()})}
    for arm in ARMS:
        for metric in ('attempts', 'valid_new', 'cnn_success', 'vina_success', 'dual_success'):
            total = scopes['all_30']['arms'][arm][metric]
            assert total == sum(s['arms'][arm][metric] for s in per_case.values())
            assert total == sum(s['arms'][arm][metric] for s in per_target.values())
            assert total == sum(scopes[s]['arms'][arm][metric] for s in ('previous_selection_10', 'previous_additional_20'))
        for metric in ('attempts', 'valid_new', 'cnn_success', 'scored'):
            assert scopes['all_30']['arms'][arm][metric] == original_accounting['scopes']['all_30']['arms'][arm][metric]
    assert scopes['all_30']['arms']['zero']['dual_success'] == 360
    assert scopes['all_30']['arms']['x0p02_h0p6']['dual_success'] == 489
    result = dict(complete=True, created_utc=now(), total_attempts=len(records), scopes=scopes,
        per_case=per_case, per_target=per_target,
        frozen_cnn_selection=read(RUN / 'selection_freeze.json')['selected_arm'],
        definition='Strict new-design task pass AND complete 22-check PoseBusters pass AND Vina < case reference AND CNN affinity > case reference, divided by all attempts.',
        interpretation=['Vina is evaluated after generation and the original CNN-only selection freeze.',
            'Both new arms are reported without score-based filtering or feedback to generation.',
            'Fixed-pose score_only; original prepared molecules and original CNN/PoseBusters scores are unchanged.',
            'These 30 previously examined development cases are not an independent holdout.'])
    write(OUT / 'records.json', records)
    write(OUT / 'references.json', references)
    write(OUT / 'accounting.json', result)
    csv_rows = [dict(level=level, scope=name, arm=arm, **m)
                for level, mapping in [('scope', scopes), ('case', per_case), ('target', per_target)]
                for name, s in mapping.items() for arm, m in s['arms'].items()]
    with (OUT / 'accounting.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    lines = ['CNN 引导全量实验：Vina + CNN 双指标评估',
             '双改善 = 严格任务、新设计、完整22项几何检查均通过，且 Vina 低于参考、CNN affinity 高于参考。',
             '每组分母固定为全部3000次尝试（30例×100次），含所有生成/重建/任务/几何失败。',
             'Vina仅在生成后评估；不参与本轮生成引导。沿用固定构象评分，不进行对接或坐标优化。', '']
    for name, s in scopes.items():
        lines.extend([name, '设置 | 合格新分子 | CNN改善 | Vina改善 | 双改善 | 分母'])
        for arm, m in s['arms'].items():
            lines.append(f"{arm} | {m['valid_new']} | {m['cnn_success']} | {m['vina_success']} | {m['dual_success']} ({m['dual_success_rate']:.2%}) | {m['attempts']}")
        lines.append('')
    lines.append('相对原最佳CNN基线的配对双改善变化：')
    for arm, m in scopes['all_30']['paired_vs_cnn_baseline'].items():
        lines.append(f"{arm}: 新增 {m['dual_gained']}，失去 {m['dual_lost']}，净变化 {m['dual_net']:+d}，{m['delta_percentage_points']:+.2f} 个百分点。")
    lines.extend(['', '原CNN规则选中方案保持为 cnn_local_geometry_guard；未根据本次Vina结果重选。',
                  '本轮为已考察开发集结果；不是独立留出集验证。'])
    (OUT / 'accounting.txt').write_text('\n'.join(lines) + '\n')
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'evaluation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status('preparing')
    records, source_rows = read(RUN / 'records.json'), read(SOURCE / 'records.json')
    refs, clean_refs = read(SOURCE / 'references.json'), read(RUN / 'references.json')
    assert read(RUN / 'audit.json')['passed'] and read(SOURCE / 'audit.json')['passed']
    assert read(RUN / 'tasks.json') == read(SOURCE / 'tasks.json')
    assert len(records) == len({key(r) for r in records}) == 12000
    tasks = {t['case_id']: t for t in read(RUN / 'tasks.json')}
    for case, s in clean_refs.items():
        assert all(refs[case][k] == v for k, v in s.items())
    source_by = {key(r): r for r in source_rows}
    jobs = []
    for row in records:
        if row['arm'] in ARMS[:2]:
            old = source_by[key(row)]
            for field in ('seed', 'path', 'reconstructed', 'raw_audit', 'noise_tape_sha256', 'raw_state_sha256'):
                assert row.get(field) == old.get(field), (key(row), field)
            assert all(old.get('raw_score', {}).get(k) == v for k, v in row.get('raw_score', {}).items()), key(row)
            if row.get('raw_score', {}).get('status') == 'scored':
                row['raw_score']['vina_kcal_mol'] = old['raw_score']['vina_kcal_mol']
                row['external_vina_evaluation'] = dict(status='reused', source_records=str(SOURCE / 'records.json'))
        elif row.get('raw_score', {}).get('status') == 'scored':
            path = Path(row['path']) / 'judge/prepared.sdf'
            protein = tasks[row['case_id']]['protein_path']
            assert sha(path) == row['raw_score']['input_sha256']
            jobs.append(dict(arm=row['arm'], case_id=row['case_id'], attempt=row['attempt'],
                input_path=str(path), input_sha256=sha(path), protein_path=protein, protein_sha256=sha(protein)))
    assert len(jobs) == 3762
    source_files = [Path(__file__).resolve(), GNINA, ROOT / 'results/selection_rules_v3/geometry_schema.json']
    source_files += [RUN / f for f in ('records.json', 'references.json', 'tasks.json', 'audit.json',
                    'manifest.json', 'execution_manifest.finalize_v1.json', 'accounting.json', 'selection.json', 'selection_freeze.json')]
    source_files += [SOURCE / f for f in ('records.json', 'references.json', 'tasks.json', 'audit.json')]
    source_hashes = {str(p): sha(p) for p in source_files}
    manifest_path = OUT / 'manifest.json'
    manifest = dict(created_utc=now(), authorization='User: 我的意思是不让你用vina做引导，不是不评vina，你现在去评双指标改善率',
        source_sha256=source_hashes, attempts_per_arm=3000, arms=ARMS, new_scores=len(jobs), jobs=jobs,
        scoring='GNINA 1.1 --scoring vina --seed 20260930 --cpu 2 --no_gpu --cnn_scoring none --score_only',
        validation='First scored attempt per available case in each new arm compared exactly with original full command; first available CNN control and all 30 references compared exactly with frozen Vina. Cases with zero strict-task passes retain all attempts in denominators.',
        denominator='All 3000 attempts per arm; strict inequalities against the same per-case reference.',
        selection_freeze=read(RUN / 'selection_freeze.json'), generation_uses_vina=False,
        evaluation_only=True, workers=WORKERS)
    if manifest_path.exists():
        previous = read(manifest_path)
        assert previous['source_sha256'] == source_hashes and previous['jobs'] == jobs
    else:
        write(manifest_path, manifest)
    status('validating', completed=0)
    validation = validate(jobs, source_rows, refs, tasks)
    scored = {}
    status('scoring', completed=0, total=len(jobs))
    with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = {pool.submit(score, j): j for j in jobs}
        for f in futures.as_completed(pending):
            result = f.result()
            scored[key(result)] = result
            if len(scored) % 100 == 0 or len(scored) == len(jobs):
                status('scoring', completed=len(scored), total=len(jobs),
                       failures=sum(s['status'] != 'scored' for s in scored.values()))
    failures = [s for s in scored.values() if s['status'] != 'scored']
    write(OUT / 'score_failures.json', failures)
    assert not failures, f'{len(failures)} unresolved scores; rerun to retry'
    for row in records:
        if key(row) in scored:
            s = scored[key(row)]
            row['raw_score']['vina_kcal_mol'] = s['vina_kcal_mol']
            row['external_vina_evaluation'] = dict(status='scored', input_sha256=s['input_sha256'],
                score_path=str(OUT / 'scores' / row['arm'] / row['case_id'] / f"{row['attempt']:04d}" / 'score.json'))
    status('accounting', completed=len(scored), total=len(jobs))
    result = summarize(records, refs, tasks, read(RUN / 'accounting.json'))
    assert all(sha(path) == expected for path, expected in source_hashes.items()), 'Frozen source changed'
    assert all(sha(j['input_path']) == j['input_sha256'] and sha(j['protein_path']) == j['protein_sha256'] for j in jobs)
    write(OUT / 'audit.json', dict(passed=True, completed_utc=now(), original_sources_unchanged=True,
        all_scoring_inputs_unchanged=True, all_attempts_retained=True, new_scores=len(scored), score_failures=0,
        command_equivalence_checks=validation['exact_vina_matches'], original_validity_cnn_counts_unchanged=True,
        original_control_dual_counts_reproduced=True, no_generation_or_selection_modified=True,
        source_sha256=source_hashes, output_sha256={f: sha(OUT / f) for f in
             ('manifest.json', 'validation.json', 'records.json', 'references.json', 'accounting.json', 'accounting.csv', 'accounting.txt')}))
    status('complete', completed=len(scored), total=len(jobs), score_failures=0,
           arms=result['scopes']['all_30']['arms'], report=str(OUT / 'accounting.txt'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=repr(error))
        raise
