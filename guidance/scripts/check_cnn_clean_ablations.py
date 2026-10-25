"""Derivative and full-trajectory gates; technical outputs never enter efficacy counts."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
import json
import time
import types
import numpy as np
import torch
from rdkit import Chem, RDLogger
from tasks import ROOT, audit, read_mol
from resume_selected_guidance_full_case_shards import read, save, sha, now, batch, TRAJECTORIES, hardware
from cnn_clean_settings import SOURCE, CHECKS, BASELINE, SETTINGS, element_strength
from cnn_local_geometry import restraints, energy
from autofragdiff_cnn_clean_hook import CleanEngine, CleanHook
from autofragdiff_direct_hook import DirectHook
from score_cnn_clean import score_one, assert_clean

FILES = ('cnn_clean_settings.py', 'cnn_local_geometry.py', 'autofragdiff_cnn_clean_hook.py',
         'score_cnn_clean.py', 'check_cnn_clean_ablations.py', 'autofragdiff_direct_hook.py',
         'gnina_soft_element_worker.py', 'gnina_soft_element_guidance.py')


def paired_step(self, *args, **kwargs):
    """Execute the frozen and new equations on identical oracle/noise outputs.

    GPU CNN backward uses kernels whose roundoff can accumulate over 500 steps.
    Sharing oracle outputs isolates implementation equivalence from that noise;
    this wrapper is used only in technical baseline replays, never efficacy runs.
    """
    if self.configuration['name'] != 'x0p02_h0p6':
        return CleanHook.step(self, *args, **kwargs)
    phase = [0]; cached = {}; positions = {}
    original_forward = self.edm.dynamics.forward
    original_noise = self.edm.sample_combined_position_features_noise
    original_gradient = self.client.joint_gradient

    def shared(name, function):
        cached[name] = []; positions[name] = 0
        def call(*a, **kw):
            if phase[0] == 0:
                value = function(*a, **kw)
                signature = (a[1].copy(), a[2].copy()) if name == 'gradient' else None
                cached[name].append((value, signature))
                return value
            index = positions[name]; positions[name] += 1
            value, signature = cached[name][index]
            if signature is not None:
                assert np.array_equal(a[1], signature[0]) and np.array_equal(a[2], signature[1])
            return value
        return call

    self.edm.dynamics.forward = shared('forward', original_forward)
    self.edm.sample_combined_position_features_noise = shared('noise', original_noise)
    self.client.joint_gradient = shared('gradient', original_gradient)
    try:
        native = DirectHook.step(self, *args, **kwargs)
        phase[0] = 1
        new = CleanHook.step(self, *args, **kwargs)
        assert all(positions[k] == len(v) for k, v in cached.items())
        assert torch.equal(native, new), ('equation mismatch', float(torch.max(torch.abs(native-new))))
        self.exact_paired_steps += 1
        return new
    finally:
        self.edm.dynamics.forward = original_forward
        self.edm.sample_combined_position_features_noise = original_noise
        self.client.joint_gradient = original_gradient


def representative_tasks():
    result = []; groups = set()
    for task in read(SOURCE / 'tasks.json'):
        if task['target_group'] not in groups: result.append(task); groups.add(task['target_group'])
    assert len(result) == 7
    return result


def cpu_checks():
    rows = []; random = np.random.default_rng(20261023)
    for task in representative_tasks():
        mol = read_mol(task['reference_path'])
        mol = Chem.RenumberAtoms(mol, task['fixed_reference_indices']+task['removed_reference_indices'])
        nc = task['fixed_heavy_atoms']; xyz = mol.GetConformer().GetPositions().copy()
        xyz[nc:] += random.normal(size=xyz[nc:].shape)*.35
        terms = restraints(mol, nc); assert terms['bonds'] and terms['angles']
        x = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        loss = energy(x, nc, terms); grad, = torch.autograd.grad(loss, x)
        g = grad.detach().numpy(); assert np.isfinite(g).all() and not np.count_nonzero(g[:nc])
        direction = random.normal(size=xyz.shape); direction[:nc] = 0; direction /= np.linalg.norm(direction)
        eps = 1e-5
        finite = float((energy(torch.tensor(xyz+eps*direction), nc, terms)-energy(torch.tensor(xyz-eps*direction), nc, terms))/(2*eps))
        analytic = float((g*direction).sum())
        assert abs(finite-analytic) < 2e-5*(1+abs(finite)), (task['case_id'], finite, analytic)
        after = float(energy(torch.tensor(xyz-1e-4*g/max(1., np.linalg.norm(g))), nc, terms))
        assert after <= float(loss.detach())+1e-10
        rows.append(dict(case_id=task['case_id'], derivative_error=abs(finite-analytic),
                         penalty_before=float(loss.detach()), penalty_after=after, core_gradient_zero=True,
                         bonds=len(terms['bonds']), angles=len(terms['angles'])))
    taper = SETTINGS['cnn_element_late_taper']
    assert element_strength(taper, 499) == element_strength(taper, 100) == .6
    assert element_strength(taper, 0) == .2
    assert np.allclose([element_strength(taper, s) for s in range(100)], np.linspace(.2, .596, 100))
    for config in [BASELINE, SETTINGS['cnn_local_geometry_guard']]:
        assert all(element_strength(config, s) == .6 for s in range(500))
    # A forbidden field must be rejected rather than silently entering the selector.
    try: assert_clean({'raw_score': {'vina_kcal_mol': -8.}})
    except AssertionError: pass
    else: raise AssertionError('withheld metric accepted')
    CHECKS.mkdir(exist_ok=True)
    save(CHECKS / 'cpu_checks.json', dict(passed=True, geometry_derivatives=rows,
         taper_boundaries_passed=True, forbidden_field_rejected=True,
         source_sha256={name: sha(ROOT / 'scripts' / name) for name in FILES}, completed_utc=now()))


def gpu_checks(gpu):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
    out = CHECKS / f'gpu{gpu}'; out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'checks.json').exists()
    source_hashes = {name: sha(ROOT / 'scripts' / name) for name in FILES}
    cpu = read(CHECKS / 'cpu_checks.json'); assert cpu['passed'] and cpu['source_sha256'] == source_hashes
    old = {(r['arm'], r['case_id'], r['attempt']): r for r in read(SOURCE / 'records.json')}
    engine = None; state = dict(gpu=gpu, pid=os.getpid(), started_utc=now())
    outputs_count = 0; replay = []; changes = []; stored = []

    def update(stage, **extra):
        state.update(stage=stage, updated_utc=now(), **extra)
        save(out / 'status.json', state); print(json.dumps(state), flush=True)

    def generate(task, config, name):
        nonlocal outputs_count
        engine.hook.configure(config, out / name)
        attempts, seeds, noise = batch(task, 0)
        with torch.no_grad(): outputs, xyz, numbers, metadata = engine.generate(task, attempts, seeds, noise)
        outputs_count += 2; nc = task['fixed_heavy_atoms']
        with np.load(TRAJECTORIES / 'arms/zero/sampling_trace' / f"{task['case_id']}_0000.npz") as native:
            assert np.array_equal(xyz[:, :nc], native['xyz'][:, :nc])
            assert np.array_equal(numbers[:, :nc], native['atomic_numbers'][:, :nc])
        assert metadata['noise_draws'] == 502 and metadata['core_max_normalized_error'] == 0
        for attempt, seed in zip(attempts, seeds):
            base = old['zero', task['case_id'], attempt]
            assert base['seed'] == seed and base['noise_tape_sha256'] == metadata['noise_tape_sha256']
        return outputs, xyz, numbers, metadata

    try:
        update('loading'); engine = CleanEngine()
        engine.hook.exact_paired_steps = 0
        engine.hook.step = types.MethodType(paired_step, engine.hook)
        tasks = representative_tasks()
        zero = dict(BASELINE, name='zero', lambda_x=0., lambda_h=0., taper_end=0.)
        outputs, xyz, numbers, meta = generate(tasks[0], zero, 'zero')
        with np.load(TRAJECTORIES / 'arms/zero/sampling_trace' / f"{tasks[0]['case_id']}_0000.npz") as native:
            assert np.array_equal(numbers, native['atomic_numbers']) and np.max(np.abs(xyz-native['xyz'])) < .001
        assert not engine.hook.pending_events
        for task in tasks:
            update('baseline_replay', active_case=task['case_id'], completed_cases=len(replay))
            outputs, xyz, numbers, meta = generate(task, BASELINE, f"baseline/{task['case_id']}")
            base = old['x0p02_h0p6', task['case_id'], 0]
            path = __import__('pathlib').Path(base['path']).parents[1] / 'sampling_trace' / f"{task['case_id']}_0000.npz"
            with np.load(path) as previous:
                error = float(np.max(np.abs(xyz-previous['xyz'])))
                assert np.array_equal(numbers, previous['atomic_numbers']), ('historical elements', task['case_id'])
            # Do not relax the old coordinate tolerance and call it passed.
            # Record its outcome explicitly; exact per-step equivalence is a separate gate.
            np.savez_compressed(out / f"baseline/{task['case_id']}/trajectory.npz", xyz=xyz, atomic_numbers=numbers)
            for i, (mol, _) in enumerate(outputs): assert audit(mol, task) == old['x0p02_h0p6', task['case_id'], i]['raw_audit']
            assert len(engine.hook.pending_traces) == 500 and len(engine.hook.pending_events) == 1000
            assert all(e['lambda_h_effective'] == .6 and e['geometry']['status'] == 'inactive' for e in engine.hook.pending_events)
            assert engine.hook.exact_paired_steps == 500*(len(replay)+1)
            replay.append(dict(case_id=task['case_id'], coordinate_error_A=error,
                               historical_coordinate_tolerance_passed=error < .001,
                               exact_native_vs_new_equation_steps=500, elements_audits_noise_equal=True))
            if task == tasks[0]:
                np.savez_compressed(out / 'baseline_trajectory.npz', xyz=xyz, atomic_numbers=numbers)
                scores = []
                for i, (mol, _) in enumerate(outputs):
                    if not audit(mol, task)['strict_success']: continue
                    score = score_one(mol, task['protein_path'], out / 'judge_baseline' / str(i))
                    previous = old['x0p02_h0p6', task['case_id'], i]['raw_score']
                    assert score['status'] == previous['status'] == 'scored'
                    assert score['posebusters_checks'] == previous['posebusters_checks']
                    delta = {m: score[m]-previous[m] for m in ('cnn_affinity_pK', 'cnn_pose_score')}
                    assert all(abs(v) < .001 for v in delta.values()); assert_clean(score)
                    scores.append(dict(attempt=i, cnn_deltas=delta, geometry_equal=True))
        setting = list(SETTINGS)[gpu]; config = SETTINGS[setting]; task = tasks[0]
        for repeat in range(2):
            update('ablation_replay', setting=setting, repeat=repeat)
            began = time.perf_counter()
            outputs, xyz, numbers, meta = generate(task, config, f'ablation_{repeat}')
            traces = engine.hook.pending_traces; events = engine.hook.pending_events
            assert len(events) == 1000 and len(traces) == 500
            assert sorted(t['step'] for t in traces) == list(range(500))
            nc = task['fixed_heavy_atoms']
            changed_geometry = 0; active = 0
            for tr in traces:
                dx, dh, before = tr['delta_x0_coordinates'], tr['delta_x0_features'], tr['coordinates_before_geometry']
                assert np.isfinite(dx).all() and np.isfinite(dh).all()
                assert not np.count_nonzero(dx[:, :nc]) and not np.count_nonzero(dh[:, :nc])
                assert np.linalg.norm(dx, axis=-1).max() <= .02+1e-7
                assert np.linalg.norm(dh, axis=-1).max() <= element_strength(config, tr['step'])+1e-7
                if tr['step'] >= 100 or not config['geometry_weight']: assert np.array_equal(dx, before)
                changed_geometry += bool(np.count_nonzero(dx-before))
            for e in events:
                assert e['lambda_h_effective'] == element_strength(config, e['step'])
                assert e['joint_model_forward_calls'] == e['joint_backward_calls'] == 1
                assert e['scored_element_substitution_candidates'] == 0
                active += e['geometry']['status'] == 'applied'
            assert bool(changed_geometry) == bool(config['geometry_weight'])
            np.savez_compressed(out / f'ablation_{repeat}/trajectory.npz', xyz=xyz, atomic_numbers=numbers,
                                dx=np.stack([t['delta_x0_coordinates'] for t in traces]),
                                dh=np.stack([t['delta_x0_features'] for t in traces]))
            save(out / f'ablation_{repeat}/events.json', events)
            changes.append(dict(seconds=time.perf_counter()-began, changed_geometry_steps=changed_geometry,
                                active_geometry_events=active, audits=[audit(m, task) for m, _ in outputs]))
            stored.append((xyz.copy(), numbers.copy()))
        assert np.array_equal(stored[0][1], stored[1][1])
        repeat_error = float(np.max(np.abs(stored[0][0]-stored[1][0]))); assert repeat_error < .001
        assert changes[0]['audits'] == changes[1]['audits']
        assert source_hashes == {name: sha(ROOT / 'scripts' / name) for name in FILES}
        save(out / 'checks.json', dict(passed=True, gpu=gpu, setting=setting, configuration=config,
             source_sha256=source_hashes, hardware=hardware(), zero_replay=True, baseline_replays=replay,
             exact_native_vs_new_equation_steps=engine.hook.exact_paired_steps,
             historical_coordinate_tolerance_all_passed=all(r['historical_coordinate_tolerance_passed'] for r in replay),
             clean_scoring_replays=scores, ablation_checks=changes, repeat_error_A=repeat_error,
             technical_outputs=outputs_count, efficacy_outputs=0, completed_utc=now(),
             output_sha256={str(p): sha(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.npz',)}))
        update('complete', passed=True)
    except BaseException as exc:
        update('failed', error=repr(exc)); raise
    finally:
        if engine is not None: engine.hook.close()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--cpu', action='store_true'); p.add_argument('--gpu', type=int, choices=(0, 1))
    args = p.parse_args(); RDLogger.DisableLog('rdApp.*'); torch.set_num_threads(4)
    if args.cpu: cpu_checks()
    else: assert args.gpu is not None; gpu_checks(args.gpu)


if __name__ == '__main__': main()
