"""Full-denominator development accounting and selection using CNN and validity only."""
import csv
import json
import numpy as np
from cnn_clean_settings import ARMS, SETTINGS
from score_cnn_clean import assert_clean


def good(row):
    s = row.get('raw_score', {})
    return bool(row['raw_audit'].get('strict_success') and row['raw_audit'].get('new_design')
                and s.get('status') == 'scored' and s.get('posebusters_complete') and s.get('posebusters_pass'))


def success(row, references):
    return good(row) and row['raw_score']['cnn_affinity_pK'] > references[row['case_id']]['cnn_affinity_pK']


def stats(values):
    return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None)


def scope(case_ids, records, references):
    wanted = set(case_ids); expected = {(case, i) for case in wanted for i in range(100)}
    by = {arm: {(r['case_id'], r['attempt']): r for r in records if r['arm'] == arm and r['case_id'] in wanted} for arm in ARMS}
    assert all(set(rows) == expected for rows in by.values())
    arms = {}
    for arm, rows in by.items():
        valid = [r for r in rows.values() if good(r)]
        passed = sum(success(r, references) for r in rows.values())
        arms[arm] = dict(attempts=len(rows), reconstructed=sum(r['reconstructed'] for r in rows.values()),
             strict=sum(r['raw_audit']['strict_success'] for r in rows.values()),
             scored=sum(r.get('raw_score', {}).get('status') == 'scored' for r in rows.values()),
             valid_new=len(valid), valid_new_rate=len(valid)/len(rows),
             cnn_success=passed, cnn_success_rate=passed/len(rows),
             score_failures=sum(r.get('raw_score', {}).get('status') == 'failed' for r in rows.values()),
             missing_geometry=sum(r.get('raw_score', {}).get('status') == 'scored' and not r['raw_score'].get('posebusters_complete') for r in rows.values()),
             cnn_affinity_valid_new=stats([r['raw_score']['cnn_affinity_pK'] for r in valid]))
    paired = {}; base = by['x0p02_h0p6']
    for arm in SETTINGS:
        shared = [k for k in sorted(expected) if good(base[k]) and good(by[arm][k])]
        paired[arm] = dict(both_valid_new=len(shared),
             cnn_delta=stats([by[arm][k]['raw_score']['cnn_affinity_pK']-base[k]['raw_score']['cnn_affinity_pK'] for k in shared]),
             valid_gained=sum(good(by[arm][k]) and not good(base[k]) for k in expected),
             valid_lost=sum(good(base[k]) and not good(by[arm][k]) for k in expected),
             cnn_success_gained=sum(success(by[arm][k], references) and not success(base[k], references) for k in expected),
             cnn_success_lost=sum(success(base[k], references) and not success(by[arm][k], references) for k in expected))
        assert paired[arm]['cnn_success_gained']-paired[arm]['cnn_success_lost'] == arms[arm]['cnn_success']-arms['x0p02_h0p6']['cnn_success']
    return dict(cases=len(wanted), attempts_per_arm=len(expected), arms=arms, paired_vs_cnn_baseline=paired)


def choose(arms):
    assert_clean(arms)
    baseline = arms['x0p02_h0p6']
    rows = [dict(arm=arm, valid_new=arms[arm]['valid_new'], cnn_success=arms[arm]['cnn_success'],
                 eligible=arms[arm]['valid_new'] >= baseline['valid_new'] and arms[arm]['cnn_success'] > baseline['cnn_success'])
            for arm in SETTINGS]
    eligible = sorted((r for r in rows if r['eligible']), key=lambda r: (-r['cnn_success'], r['arm']))
    return dict(selected_arm=eligible[0]['arm'] if eligible else 'x0p02_h0p6', rows=rows,
                rule='Require valid-new count >= paired CNN baseline; maximize strictly improved CNN-success count; lexicographic arm name breaks ties.',
                evidence='Existing development cases; this is not independent validation.', external_evaluation='withheld')


def write_accounting(run, records, tasks, references):
    assert_clean(records); assert_clean(references)
    assert len(records) == len({(r['arm'], r['case_id'], r['attempt']) for r in records}) == 12000
    manifest = json.loads((run / 'manifest.json').read_text())
    ids = [t['case_id'] for t in tasks]; old_selection = set(manifest['previous_selection_case_ids'])
    scopes = {name: scope(cases, records, references) for name, cases in (
        ('all_30', ids), ('previous_selection_10', [c for c in ids if c in old_selection]),
        ('previous_additional_20', [c for c in ids if c not in old_selection]))}
    per_case = {case: scope([case], records, references) for case in ids}
    per_target = {group: scope([t['case_id'] for t in tasks if t['target_group'] == group], records, references)
                  for group in sorted({t['target_group'] for t in tasks})}
    for arm in ARMS:
        for metric in ('attempts', 'valid_new', 'cnn_success'):
            expected = scopes['all_30']['arms'][arm][metric]
            assert expected == sum(s['arms'][arm][metric] for s in per_case.values())
            assert expected == sum(s['arms'][arm][metric] for s in per_target.values())
            assert expected == sum(scopes[s]['arms'][arm][metric] for s in ('previous_selection_10', 'previous_additional_20'))
    assert scopes['all_30']['arms']['x0p02_h0p6']['valid_new'] == 1341
    assert scopes['all_30']['arms']['x0p02_h0p6']['cnn_success'] == 903
    selection = choose(scopes['all_30']['arms'])
    result = dict(complete=True, new_outputs=6000, reused_outputs=6000, total_outputs=12000,
                  scopes=scopes, per_case=per_case, per_target=per_target, selection=selection,
                  target_macro_rates={a: {k:float(np.mean([s['arms'][a][k] for s in per_target.values()]))
                     for k in ('valid_new_rate', 'cnn_success_rate')} for a in ARMS},
                  interpretation=['All attempts remain in denominators, including reconstruction, task and scoring failures.',
                    'Only complete 22-check geometry passes count as valid.',
                    'Selection uses CNN and geometry only; the external affinity evaluator is withheld.',
                    'All 30 cases have been examined previously; fresh cases are needed for independent validation.'])
    assert_clean(result)
    (run / 'accounting.json').write_text(json.dumps(result, indent=2)+'\n')
    (run / 'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    rows = []
    for level, mapping in [('scope', scopes), ('case', per_case), ('target', per_target)]:
        for name, s in mapping.items():
            for arm, m in s['arms'].items():
                rows.append(dict(level=level, name=name, arm=arm, **{k:v for k,v in m.items() if not isinstance(v,dict)},
                                 cnn_valid_mean=m['cnn_affinity_valid_new']['mean'], cnn_valid_median=m['cnn_affinity_valid_new']['median']))
    with (run / 'accounting.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['CNN 单目标全量消融：开发核算', '每组30例 × 100次；新增两组共6000次，复用零引导与CNN基线共6000次。',
             '开发指标：合格且CNN高于参考／全部尝试；外部亲和力指标保持未解封。', '']
    for name, s in scopes.items():
        lines += [name, '设置 | 合格新分子 | 合格且CNN优于参考 | 总尝试']
        for arm, m in s['arms'].items():
            lines.append(f"{arm} | {m['valid_new']} ({m['valid_new_rate']:.2%}) | {m['cnn_success']} ({m['cnn_success_rate']:.2%}) | {m['attempts']}")
        lines.append('')
    lines += [f"按冻结规则选中：{selection['selected_arm']}", '本轮是已有开发集结果，不能据此宣称独立联合指标已达标。']
    (run / 'accounting.txt').write_text('\n'.join(lines)+'\n')
    return result
