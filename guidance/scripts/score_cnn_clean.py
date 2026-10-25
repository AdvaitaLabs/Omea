"""Frozen fixed-heavy-atom CNN/PoseBusters judging, with no empirical score retained.

The pinned GNINA score_only executable still computes its usual outputs. Only
the two CNN lines are parsed or saved; its empirical affinity line is discarded.
The development accounting and selector therefore receive no withheld metric.
"""
import json
import re
import subprocess
import time
import numpy as np
from score_extra_results import prepare, GNINA
from tasks import write_sdf

SCORE_KEYS = ('status', 'mode', 'preparation', 'input_sha256', 'gnina_seconds',
              'cnn_affinity_pK', 'cnn_pose_score', 'posebusters_checks', 'posebusters_pass',
              'posebusters_seconds', 'posebusters_error', 'error', 'wall_seconds',
              'legacy_posebusters_pass', 'posebusters_complete', 'missing_checks',
              'scoring_serialization')


def assert_clean(value):
    if isinstance(value, dict):
        for key, item in value.items():
            assert not any(token in key.lower() for token in ('vina', 'dual_', 'delta_affinity')), key
            assert_clean(item)
    elif isinstance(value, (list, tuple)):
        for item in value: assert_clean(item)


def project_score(score):
    result = {key: score[key] for key in SCORE_KEYS if key in score}
    assert_clean(result)
    return result


def score_one(mol, protein, out):
    import hashlib
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / 'score.json'
    if result_path.exists():
        result = json.loads(result_path.read_text()); assert_clean(result); return result
    began = time.perf_counter(); result = dict(status='pending', mode='fixed_pose_score_only')
    if mol.HasProp('scoring_serialization'):
        result['scoring_serialization'] = json.loads(mol.GetProp('scoring_serialization'))
    try:
        hmol, prep = prepare(mol); write_sdf(out / 'prepared.sdf', [hmol]); result['preparation'] = prep
        result['input_sha256'] = hashlib.sha256((out / 'prepared.sdf').read_bytes()).hexdigest()
        # Same judge and options as the audited controls; empirical affinity is not an output of this wrapper.
        command = [str(GNINA), '-r', str(protein), '-l', str(out / 'prepared.sdf'), '--scoring', 'vina',
                   '--seed', '20260930', '--cpu', '2', '--no_gpu', '--cnn_scoring', 'rescore', '--score_only']
        t = time.perf_counter()
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        result['gnina_seconds'] = time.perf_counter()-t
        if proc.returncode: raise RuntimeError(f'GNINA exit {proc.returncode}; raw empirical-score output not retained')
        lines = []
        for name, key in [('CNNaffinity', 'cnn_affinity_pK'), ('CNNscore', 'cnn_pose_score')]:
            match = re.search(rf'^{name}:\s*([-+0-9.eE]+)', proc.stdout, re.M)
            if match is None: raise ValueError(f'GNINA CNN metric missing: {name}')
            result[key] = float(match.group(1)); assert np.isfinite(result[key])
            lines.append(f'{name}: {match.group(1)}')
        (out / 'cnn.log').write_text('\n'.join(lines)+'\n')
        result['status'] = 'scored'
        try:
            from posebusters import PoseBusters
            t = time.perf_counter(); pb = PoseBusters(config='dock', max_workers=0)
            frame = pb.bust(mol_pred=hmol, mol_cond=str(protein), full_report=False)
            checks = {k: bool(v) for k, v in frame.iloc[0].items() if isinstance(v, (bool, np.bool_))}
            if not checks: raise ValueError('no PoseBusters checks returned')
            result['posebusters_checks'] = checks; result['posebusters_pass'] = all(checks.values())
            result['posebusters_seconds'] = time.perf_counter()-t
            frame.to_csv(out / 'posebusters.csv')
        except Exception as exc: result['posebusters_error'] = repr(exc)
    except subprocess.TimeoutExpired:
        result.update(status='failed', error='GNINA timed out after 120 seconds; raw output not retained')
    except Exception as exc: result.update(status='failed', error=repr(exc))
    result['wall_seconds'] = time.perf_counter()-began
    assert_clean(result)
    temporary = result_path.with_suffix('.tmp'); temporary.write_text(json.dumps(result, indent=2)); temporary.replace(result_path)
    return result
