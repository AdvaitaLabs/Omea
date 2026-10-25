"""Two-case, single-seed-root strength screen; one output per paired attempt.

Reuse verified native-reconstruction controls. Freeze the original continuous
GNINA gradient implementation; only coordinate/feature strengths vary. The
final decoder uses native reconstruction without extra core graph restoration.
"""
import os
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
import copy
import hashlib
import json
import shutil
import time
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from tasks import ROOT, audit
from autofragdiff_direct_hook import StrengthEngine, make_noise, NUMBERS
from autofragdiff_strength_hook import NoiseTape
from autofragdiff_local_regeneration import sample_restart
from reanalyze_geometry_stereo import normalize
from evaluate_direct_guidance import score_arm, summarize
from score_extra_results import score_one
from run_direct_guidance_experiment import SOURCES

SOURCE = ROOT / 'results/direct_guidance_native_reconstruction_scale30_v1'
TRAJECTORIES = ROOT / 'results/direct_guidance_joint_strength_scale30_v1'
CASE_IDS = ('7AKI-bio1_RJQ_A_202__a001_r000', '3GTA-bio2_851_B_1__a002_r009')
ATTEMPTS = 100
BATCH = 2
SEED_ROOT = 2026101203


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def configurations():
    timing = dict(last_fraction=1., interval=1, schedule='flat', temperature=.2)
    result = [dict(name='zero', lambda_x=0., lambda_h=0., reuse_arm='zero', **timing)]
    result += [dict(name='joint_'+str(h).replace('.', 'p'), lambda_x=.005,
                    lambda_h=h, reuse_arm='joint_'+str(h).replace('.', 'p'), **timing)
               for h in [.2, .4, .6]]
    # Run the central coordinate-strength comparison first.
    result += [dict(name='x'+str(x).replace('.', 'p')+'_h'+str(h).replace('.', 'p'),
                    lambda_x=x, lambda_h=h, reuse_arm=None, **timing)
               for h in [.4, .2, .6] for x in [.01, .02]]
    result += [dict(name='coordinate_only_x0p01', lambda_x=.01, lambda_h=0., reuse_arm=None, **timing),
               dict(name='element_only_h0p4', lambda_x=0., lambda_h=.4, reuse_arm=None, **timing)]
    assert len(result) == 12 and sum(c['reuse_arm'] is None for c in result) == 8
    return result


class NativeFinalEngine(StrengthEngine):
    """Same sampler and hook; final chemistry matches the corrected controls."""
    def generate(self, task, attempts, seeds, noise):
        inputs, center, core = self.fresh_inputs(task, len(attempts))
        self.hook.context(task, center, core, attempts, seeds)
        tape = NoiseTape(noise)
        def draw(_edm, n_samples, n_nodes, mask):
            return tape.draw(n_samples, n_nodes, mask)
        self.edm.sample_combined_position_features_noise = types.MethodType(draw, self.edm)
        try:
            x, h, metadata = sample_restart(self.edm, inputs, 'full', self.edm.T)
            assert tape.position == self.edm.T + 2 == 502
        finally:
            self.edm.sample_combined_position_features_noise = self.original_noise
        xyz = x.cpu().numpy() + center
        numbers = NUMBERS[h.argmax(-1).cpu().numpy()]
        outputs = []
        for pos, atoms in zip(xyz, numbers):
            try:
                mol = self.reconstruct(pos.tolist(), atoms.tolist())
                Chem.SanitizeMol(mol)
                mol = normalize(mol)[0]
                assert [a.GetAtomicNum() for a in mol.GetAtoms()] == atoms.tolist()
                for i, point in enumerate(np.round(mol.GetConformer().GetPositions(), 4)):
                    mol.GetConformer().SetAtomPosition(i, point)
                assert np.max(np.abs(mol.GetConformer().GetPositions()-pos)) < .000051
                outputs.append((mol, None))
            except AssertionError:
                raise
            except Exception as exc:
                outputs.append((None, repr(exc)))
        return outputs, xyz, numbers, dict(**metadata, noise_tape_sha256=tape.sha256,
                                          noise_draws=tape.position)


def good(row):
    return bool(row['raw_audit'].get('strict_success') and row['raw_audit'].get('new_design')
                and row.get('raw_score', {}).get('status') == 'scored'
                and row['raw_score'].get('posebusters_pass'))


def parameter_selection(result, records):
    """Select one global setting; never choose a different output per seed."""
    by = {}
    for r in records:
        by.setdefault(r['arm'], {})[r['case_id'], r['attempt']] = r
    current = result['arms']['joint_0p4']['metrics']
    rows = []
    for name, arm in result['arms'].items():
        if name in ('zero', 'joint_0p4'):
            continue
        keys = sorted(k for k in by[name] if all(good(by[a][k]) for a in ('zero', 'joint_0p4', name)))
        row = dict(arm=name, common_valid_triplets=len(keys),
                   valid_new=arm['metrics']['valid_new'], dual_vs_reference=arm['metrics']['dual_vs_reference'])
        for metric, label in [('vina_kcal_mol', 'vina'), ('cnn_affinity_pK', 'cnn')]:
            delta = [by[name][k]['raw_score'][metric] - by['joint_0p4'][k]['raw_score'][metric] for k in keys]
            row[label+'_mean_delta_vs_current'] = float(np.mean(delta)) if delta else None
            row[label+'_median_delta_vs_current'] = float(np.median(delta)) if delta else None
        row['eligible'] = bool(keys and row['valid_new'] >= current['valid_new']
                               and row['dual_vs_reference'] >= current['dual_vs_reference']
                               and row['vina_mean_delta_vs_current'] <= 0
                               and row['cnn_mean_delta_vs_current'] >= 0
                               and (row['vina_mean_delta_vs_current'] < 0 or row['cnn_mean_delta_vs_current'] > 0))
        rows.append(row)
    eligible = sorted((r for r in rows if r['eligible']),
                      key=lambda r:(r['vina_mean_delta_vs_current'], -r['cnn_mean_delta_vs_current'], r['arm']))
    return dict(status='development_parameter_screen', selected_global_arm=eligible[0]['arm'] if eligible else 'joint_0p4',
                rule='Preserve valid-new and dual counts; both mean scores no worse than current 0.4 on common valid triplets; prioritize Vina then CNN; fixed arm name breaks exact ties.',
                rows=rows, future_timing_or_multiseed_jobs_launched=False,
                limitation='Two previously inspected cases and one seed root; no confirmatory or new-target claim.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='direct_guidance_parameter_screen_2case_v3')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')
    torch.set_num_threads(4)
    run = ROOT / 'results' / args.run
    configs = configurations()
    source_records = json.loads((SOURCE/'records.json').read_text())
    source_by_key = {(r['arm'],r['case_id'],r['attempt']):r for r in source_records}
    source_comparison = json.loads((SOURCE/'comparison.json').read_text())
    assert source_comparison['complete'] and json.loads((SOURCE/'audit.json').read_text())['passed']
    old_manifest = json.loads((TRAJECTORIES/'manifest.json').read_text())
    assert old_manifest['seed_root'] == SEED_ROOT and old_manifest['batch_size'] == BATCH
    taskmap = {t['case_id']:t for t in json.loads((SOURCE/'tasks.json').read_text())}
    tasks = [taskmap[k] for k in CASE_IDS]
    assert len({t['target_group'] for t in tasks}) == 2
    references_path = ROOT/'results/direct_guidance_0p4_expansion_v1/expanded_references.json'
    references = {r['case_id']:r for r in json.loads(references_path.read_text())['rows']
                  if r['model']=='reference' and r['case_id'] in CASE_IDS}
    assert len(references) == 2 and all(r['status']=='scored' for r in references.values())
    geometry = ROOT/'results/selection_rules_v3/geometry_schema.json'
    required = set(json.loads(geometry.read_text())['required_checks'])
    assert len(required) == 22
    sources = list(dict.fromkeys(SOURCES + ['run_direct_guidance_parameter_screen.py']))
    # The existing direct gradient implementation must match the frozen prior run.
    for name in SOURCES:
        assert sha(ROOT/'scripts'/name) == sha(TRAJECTORIES/'source'/name), name
    if args.resume:
        manifest = json.loads((run/'manifest.json').read_text())
        assert manifest['configurations'] == configs
        records = json.loads((run/'records.json').read_text())
        state = json.loads((run/'status.json').read_text())
        if state['stage'] == 'complete':
            print('Already complete', flush=True)
            return
        state['pid'] = os.getpid()
    else:
        run.mkdir(exist_ok=False)
        (run/'source').mkdir()
        for name in sources:
            shutil.copy2(ROOT/'scripts'/name, run/'source'/name)
        native = ROOT/'vendor/AutoFragDiff/analysis/reconstruct_mol.py'
        shutil.copy2(native, run/'source/native_reconstruct_mol.py')
        inputs = [SOURCE/'records.json', SOURCE/'comparison.json', SOURCE/'audit.json',
                  references_path, geometry, native, ROOT/'checkpoints/autofragdiff_pocket-gvp.ckpt',
                  ROOT/'vendor/AutoFragDiff/src/edm.py', ROOT/'vendor/AutoFragDiff/src/dynamics_gvp.py',
                  ROOT/'reports/guidance_replay_numerical_calibration_20261019.json']
        inputs += sorted((ROOT/'vendor/gnina-torch-guidance/gninatorch/weights').glob('*.pt'))
        for task in tasks:
            inputs += [Path(task[k]) for k in ['core_path', 'protein_path', 'reference_path']]
        manifest = dict(created_utc=now(), cases=2, target_groups=2, attempts_per_case=100,
                        outputs_per_arm=200, total_outputs=2400, new_sampling_outputs=1600,
                        reused_outputs=800, seed_root=SEED_ROOT, seed_roots=1, batch_size=BATCH,
                        case_selection='First R4 case and first R7 case from a different target group in the existing task order; input metadata only.',
                        source_run=str(SOURCE), source_trajectories=str(TRAJECTORIES),
                        configurations=configs, source_sha256={n:sha(ROOT/'scripts'/n) for n in sources},
                        input_sha256={str(p):sha(p) for p in inputs},
                        protocol=dict(pairing='Same case, attempt, seed and 502 Gaussian draws; original batch size 2.',
                                      sampling='One final native output per seed per arm; all failures retained.',
                                      gradient='Unchanged direct coordinate and continuous element-density GNINA gradients over all R atoms.',
                                      reconstruction='Native final reconstruction, coordinate-derived stereo normalization, 4-decimal coordinates; no added core graph restoration.',
                                      evaluation='Shared H-only preparation, score-only Vina/GNINA, complete 22-check PoseBusters.',
                                      constraints='Frozen core coordinates/types, checkpoint, R count, temperature, timing and clipping.',
                                      budget='No multiple seed roots, best-of-K, redraws, zero fallback or post-generation heavy-atom optimization.',
                                      next_steps='Strength screen only. No automatic timing, larger-cohort or multiseed queue.'),
                        limitations=['Development parameter screen, not an independent evaluation.',
                                     'Vina participates in parameter selection; it is not an untouched independent judge.',
                                     'Proxy scores are not measured DDG; equal output counts do not imply equal GPU compute.'])
        save(run/'manifest.json', manifest)
        save(run/'tasks.json', tasks)
        save(run/'configurations.json', configs)
        save(run/'references.json', references)
        records = []
        for cfg in configs:
            if cfg['reuse_arm'] is None:
                continue
            selected = [r for r in source_records if r['arm']==cfg['reuse_arm'] and r['case_id'] in CASE_IDS]
            assert len(selected) == 200
            for original in selected:
                r = copy.deepcopy(original)
                r.update(arm=cfg['name'], reused=True, reused_source_run=str(SOURCE),
                         reused_original_path=original['path'])
                records.append(r)
        save(run/'records.json', records)
        state = dict(pid=os.getpid(), started_utc=now(), stage='preparing', generated_outputs=0,
                     reused_outputs=800, total_outputs=2400, new_total_outputs=1600,
                     completed_arms=[c['name'] for c in configs if c['reuse_arm'] is not None])

    def update(stage, **values):
        state.update(stage=stage, updated_utc=now(), **values)
        state['generated_outputs'] = sum(not r.get('reused', False) for r in records)
        save(run/'status.json', state)
        print(json.dumps(state), flush=True)

    def integrity():
        for name, digest in manifest['source_sha256'].items():
            assert sha(ROOT/'scripts'/name) == digest, name
        for name, digest in manifest['input_sha256'].items():
            assert sha(name) == digest, name

    def summarize_screen():
        result = summarize(run, configs, records, tasks, references)
        for cfg in configs:
            if cfg['reuse_arm'] is not None:
                result['arms'][cfg['name']]['mechanism'] = dict(
                    reused=True, guidance_rerun=False, source_run=str(SOURCE),
                    source_arm=cfg['reuse_arm'], subset_outputs=200,
                    source_full_cohort_gradient_events=source_comparison['arms'][cfg['reuse_arm']]['mechanism']['guided_events'],
                    note='Metrics use only the two registered cases; no new guidance or scoring for reused controls.')
        result['scope'] = 'Two cases, one seed root, 100 attempts per case; development strength screen only.'
        save(run/'comparison.json', result)
        return result

    def batch(task, offset):
        attempts = list(range(offset, offset+BATCH))
        seeds = [(SEED_ROOT+int(hashlib.sha256(f"{task['case_id']}/{i}".encode()).hexdigest()[:8],16))%(2**32)
                 for i in attempts]
        noise = make_noise(seeds, task['fixed_heavy_atoms']+task['rgroup_heavy_atoms'])
        return attempts, seeds, noise

    base = {(r['case_id'],r['attempt']):r for r in records if r['arm']=='zero'}
    engine = None
    timing_path = run/'timing.json'
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    try:
        integrity()
        update('loading_checkpoint', active_arm=None)
        summarize_screen()
        engine = NativeFinalEngine()
        assert engine.edm.T == 500
        gate = run/'preflight_checks.json'
        if not gate.exists():
            checks = []
            for source_arm in ('zero', 'joint_0p4'):
                cfg = next(c for c in configs if c['name']==source_arm)
                engine.hook.configure(cfg, run/'preflight'/source_arm)
                for task in tasks:
                    update('preflight_replay', active_arm=source_arm, active_case=task['case_id'])
                    attempts, seeds, noise = batch(task, 0)
                    with torch.no_grad():
                        outputs, xyz, numbers, metadata = engine.generate(task, attempts, seeds, noise)
                    expected = np.load(TRAJECTORIES/'arms'/source_arm/'sampling_trace'/f"{task['case_id']}_0000.npz")
                    replay = run/'preflight'/source_arm/task['case_id']
                    replay.mkdir(exist_ok=True)
                    np.savez_compressed(replay/'sampling_trace.npz',xyz=xyz,atomic_numbers=numbers,seeds=seeds)
                    delta = float(np.max(np.abs(xyz-expected['xyz'])))
                    # Native GPU replays can accumulate FP drift beyond SDF
                    # quantization: retained calibration measured 0.0001105 A
                    # on a zero-guidance replay, with identical chemistry and
                    # all geometry checks, and <=0.00028 score drift. Coordinate
                    # agreement is a 0.001 A guard; exact chemical/task/geometry
                    # equality and <0.001 score differences remain mandatory.
                    tolerance = .001
                    assert np.array_equal(numbers, expected['atomic_numbers']), (source_arm, task['case_id'])
                    assert delta < tolerance, (source_arm, task['case_id'], delta, tolerance)
                    assert metadata['noise_tape_sha256'] == base[task['case_id'],0]['noise_tape_sha256']
                    scored_replays = []
                    for j,(mol,error) in enumerate(outputs):
                        old = source_by_key[source_arm,task['case_id'],attempts[j]]
                        actual_audit = audit(mol,task)
                        save(replay/f'audit_{attempts[j]:04d}.json',actual_audit)
                        assert actual_audit == old['raw_audit'], (source_arm,task['case_id'],attempts[j],'audit changed')
                        if actual_audit['strict_success']:
                            s = score_one(mol,task['protein_path'],replay/f'judge_{attempts[j]:04d}')
                            previous = old['raw_score']
                            assert s['status']==previous['status']=='scored'
                            assert s['posebusters_checks']==previous['posebusters_checks']
                            changes = {key:s[key]-previous[key] for key in
                                       ('vina_kcal_mol','cnn_affinity_pK','cnn_pose_score')}
                            assert all(abs(v)<.001 for v in changes.values()), changes
                            scored_replays.append(dict(attempt=attempts[j],score_deltas=changes,
                                                       all_geometry_checks_equal=True))
                    checks.append(dict(arm=source_arm, case_id=task['case_id'], attempts=attempts,
                                       max_replay_coordinate_delta_A=delta, coordinate_tolerance_A=tolerance,
                                       elements_equal=True,exact_chemical_and_task_audit_equal=True,
                                       scored_replays=scored_replays))
                    save(run/'preflight_progress.json',dict(passed_so_far=True,rows=checks))
            save(gate, dict(passed=True, cases=2, technical_replayed_outputs=8,
                            replay_coordinate_tolerance_A=.001,score_delta_tolerance=.001,
                            calibration_note='v1/v2 retained, no new efficacy outputs. Independent analysis of v2 zero replays found 0.000110507 A drift with identical chemistry, task and geometry checks and max 0.00028 score difference. v3 requires exact chemical/task/geometry equality and <0.001 score agreement, alongside a 0.001 A coordinate guard. No output repair or score adjustment.',
                            rows=checks,
                            counts_as_new_efficacy_outputs=False))
        assert json.loads(gate.read_text())['passed']
        for cfg in configs:
            arm = cfg['name']
            if arm in state['completed_arms']:
                continue
            integrity()
            engine.hook.configure(cfg, run/'arms'/arm)
            batches = timing.setdefault(arm, dict(batches=[]))['batches']
            for task in tasks:
                for offset in range(0, ATTEMPTS, BATCH):
                    present = [r for r in records if r['arm']==arm and r['case_id']==task['case_id']
                               and r['attempt'] in range(offset,offset+BATCH)]
                    if present:
                        assert len(present)==BATCH
                        continue
                    update('generating', active_arm=arm, active_case=task['case_id'], active_offset=offset,
                           active_arm_generated=sum(r['arm']==arm for r in records))
                    attempts, seeds, noise = batch(task, offset)
                    began = time.perf_counter()
                    with torch.no_grad():
                        outputs, xyz, numbers, metadata = engine.generate(task, attempts, seeds, noise)
                    seconds = time.perf_counter()-began
                    folder = run/'arms'/arm/'sampling_trace'
                    folder.mkdir(exist_ok=True)
                    np.savez_compressed(folder/f"{task['case_id']}_{offset:04d}.npz", xyz=xyz,
                                        atomic_numbers=numbers, seeds=seeds)
                    for j, (mol, error) in enumerate(outputs):
                        baseline = base[task['case_id'],attempts[j]]
                        assert seeds[j] == baseline['seed']
                        assert metadata['noise_tape_sha256'] == baseline['noise_tape_sha256']
                        nc = task['fixed_heavy_atoms']
                        expected = np.load(TRAJECTORIES/'arms/zero/sampling_trace'/f"{task['case_id']}_{offset:04d}.npz")
                        assert np.array_equal(numbers[j,:nc],expected['atomic_numbers'][j,:nc])
                        assert np.array_equal(xyz[j,:nc],expected['xyz'][j,:nc])
                        path = run/'arms'/arm/task['case_id']/f'{attempts[j]:04d}'
                        path.mkdir(parents=True, exist_ok=True)
                        r = dict(arm=arm, case_id=task['case_id'], attempt=attempts[j], seed=seeds[j],
                                 reused=False, path=str(path), reconstructed=mol is not None,
                                 reconstruction_error=error, raw_audit=audit(mol,task), **metadata)
                        if mol is not None:
                            options = Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble
                            (path/'raw.rdmol').write_bytes(mol.ToBinary(options))
                            r['raw_state_sha256'] = sha(path/'raw.rdmol')
                            with Chem.SDWriter(str(path/'raw.sdf')) as writer:
                                writer.SetKekulize(False)
                                writer.write(Chem.Mol(mol))
                        save(path/'attempt.json',r)
                        records.append(r)
                    batches.append(dict(case_id=task['case_id'], offset=offset, seconds=seconds))
                    save(run/'records.json',records)
                    save(timing_path,timing)
            began = time.perf_counter()
            update('scoring', active_arm=arm, completed_scores=0)
            score_arm(run, arm, records, tasks, required, update)
            timing[arm]['score_wall_seconds'] = time.perf_counter()-began
            timing[arm]['generation_seconds'] = sum(b['seconds'] for b in batches)
            save(timing_path,timing)
            summarize_screen()
            state['completed_arms'].append(arm)
            update('arm_complete',active_arm=arm)
        update('auditing',active_arm=None)
        assert len(records) == len(configs)*200 == 2400
        assert len({(r['arm'],r['case_id'],r['attempt']) for r in records}) == 2400
        event_audits = {}
        for cfg in configs:
            if cfg['reuse_arm'] is not None:
                continue
            path = run/'arms'/cfg['name']/'events.jsonl'
            events = [json.loads(s) for s in path.read_text().splitlines()]
            assert len(events) == 200*500
            count = Counter((e['case_id'],e['attempt']) for e in events)
            assert len(count)==200 and all(n==500 for n in count.values())
            assert all(e['status']=='guided' and e['joint_model_forward_calls']==e['joint_backward_calls']==1
                       and e['scored_element_substitution_candidates']==0 for e in events)
            for trace in (run/'arms'/cfg['name']/'guidance_traces').glob('*.npz'):
                case_id = trace.name.rsplit('_',2)[0]
                nc = taskmap[case_id]['fixed_heavy_atoms']
                with np.load(trace) as d:
                    assert np.count_nonzero(d['delta_x0_coordinates'][:,:nc])==0
                    assert np.count_nonzero(d['delta_x0_features'][:,:nc])==0
                    if cfg['lambda_x']==0: assert np.count_nonzero(d['delta_x0_coordinates'])==0
                    if cfg['lambda_h']==0: assert np.count_nonzero(d['delta_x0_features'])==0
            event_audits[cfg['name']] = dict(outputs=200, verified_gradient_events=len(events))
        for r in records:
            if r['reconstructed']:
                assert sha(Path(r['path'])/'raw.rdmol') == r['raw_state_sha256']
                assert audit(Chem.Mol((Path(r['path'])/'raw.rdmol').read_bytes()),taskmap[r['case_id']]) == r['raw_audit']
        integrity()
        result = summarize_screen()
        assert result['complete']
        save(run/'selection.json',parameter_selection(result,records))
        save(run/'audit.json',dict(passed=True, seed_roots=1, cases=2, attempts_per_case=100,
                                  new_outputs=1600,reused_outputs=800,all_attempts_retained=True,
                                  native_final_reconstruction=True,core_frozen=True,
                                  no_best_of_K=True,no_new_timing_or_multiseed_jobs=True,arms=event_audits))
        update('complete',active_arm=None,completed_utc=now(),result_file=str(run/'comparison.json'))
    except BaseException as exc:
        save(run/'records.json',records)
        update('failed',error=repr(exc))
        raise
    finally:
        if engine is not None:
            engine.hook.close()


if __name__ == '__main__':
    main()
