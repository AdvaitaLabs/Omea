"""Generate a fixed raw attempt budget with the existing checkpoint adapters."""
import argparse
import datetime
import hashlib
import importlib
import json
import os
import shutil
import time
from pathlib import Path
from benchmark_io import ROOT, CODE_ROOT, INFERENCE_KEYS, dump, input_hashes, load_tasks, methods, safe_name, save_molecules, sha, source_hashes


def run_cases(generate, tasks, dest, model, attempts, batch_size, seed_root, seed_strategy):
    import random
    import numpy as np
    import torch
    from normalization import normalize
    from tasks import audit
    from rdkit import Chem

    for task in tasks:
        out = dest/task['case_id']
        if (out/'summary.json').exists():
            rows = json.loads((out/'attempts.json').read_text())
            if len(rows) != attempts or {r['attempt'] for r in rows} != set(range(attempts)):
                raise ValueError(f'Corrupt completed case: {out}')
            archived = json.loads((out/'molecules.json').read_text())
            for entry in archived:
                if sha(out/entry['file']) != entry['sha256']:
                    raise ValueError(f'Archived output changed: {out/entry["file"]}')
            continue
        # An interrupted case is replayed from its original seed. Completed
        # cases are reused; partial batches are never selected by output quality.
        out.mkdir(parents=True, exist_ok=True)
        case_seed = seed_root + int(hashlib.sha256(task['case_id'].encode()).hexdigest()[:6], 16)
        def seed(value):
            torch.manual_seed(value); np.random.seed(value % 2**32); random.seed(value)
        seed(case_seed)
        rows, molecules, times = [], [], []
        began = time.perf_counter()
        for offset in range(0, attempts, batch_size):
            count = min(batch_size, attempts-offset)
            batch_seed = case_seed+offset if seed_strategy == 'case_seed_plus_batch_offset' else case_seed
            if seed_strategy == 'case_seed_plus_batch_offset':
                seed(batch_seed)
            inputs = {key: task[key] for key in INFERENCE_KEYS}
            timer = time.perf_counter()
            try:
                generated = generate(inputs, count)
                if len(generated) != count:
                    raise RuntimeError(f'Adapter output count mismatch: {len(generated)} != {count}')
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as exc:
                import traceback
                dump(out/'runtime_error.json', dict(offset=offset, count=count,
                     error=repr(exc), traceback=traceback.format_exc()))
                raise
            seconds = time.perf_counter()-timer
            times.append(seconds)
            for j, mol in enumerate(generated):
                if mol is not None:
                    mol, _ = normalize(mol)
                row = audit(mol, task)
                row.update(attempt=offset+j, seed=batch_seed, model=model,
                           audit_version='geometry_stereo_v2', batch_seconds=seconds)
                if mol is not None:
                    mol = Chem.Mol(mol)
                    mol.SetProp('_Name', f'{model}_{task["case_id"]}_{offset+j}')
                    molecules.append((offset+j, mol))
                rows.append(row)
            dump(out/'attempts.partial.json', rows)
            print(json.dumps(dict(model=model, case=task['case_id'], completed=len(rows),
                                  attempts=attempts, strict_success=sum(r['strict_success'] for r in rows))), flush=True)
        # Archiving native graphs avoids SDF aromatic-H ambiguity. This changes
        # file storage, not graph reconstruction, sampling or failed-attempt policy.
        save_molecules(out, molecules)
        dump(out/'attempts.json', rows)
        good = [r for r in rows if r['strict_success']]
        dump(out/'summary.json', dict(model=model, case_id=task['case_id'], attempts=len(rows),
             valid=sum(r['valid'] for r in rows), strict_success=len(good),
             unique_new_designs=len({r['smiles'] for r in good if r['new_design']}),
             generation_seconds=sum(times), wall_seconds=time.perf_counter()-began,
             batch_seconds=times, runtime_failure_attempts=0))
        (out/'attempts.partial.json').unlink(missing_ok=True)
        (out/'runtime_error.json').unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(methods()), required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--tasks', default=str(ROOT/'data/benchmark_tasks.json'))
    parser.add_argument('--cohort', choices=['all', 'primary', 'single_atom'], default='all',
                        help='all=45 prepared cases; primary=30 common multi-atom cases; single_atom=14 secondary cases')
    parser.add_argument('--attempts', type=int, default=100)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--smoke', action='store_true', help='Explicitly mark a small integration test; never use it in the main table')
    parser.add_argument('--limit', type=int, help='Only available with --smoke')
    args = parser.parse_args(argv)
    if args.attempts < 1 or (not args.smoke and args.attempts != 100):
        parser.error('Formal benchmark requires exactly 100 attempts per case; use --smoke for integration checks')
    if args.limit is not None and (not args.smoke or args.limit < 1):
        parser.error('--limit requires --smoke and a positive value')
    selected = methods()[args.model]
    args.batch_size = args.batch_size or selected['batch_size']
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    safe_name(args.run)
    tasks = load_tasks(args.tasks)
    if args.cohort == 'primary':
        tasks = [task for task in tasks if task['primary']]
    elif args.cohort == 'single_atom':
        tasks = [task for task in tasks if task['stratum'] == 'single_atom']
    if not tasks:
        parser.error('No tasks remain in the selected cohort')
    if args.limit:
        tasks = tasks[:args.limit]
    if args.model == 'FLOWR.root' and any(t['pdb_id'] == '1RVX' for t in tasks):
        parser.error('FLOWR.root does not support 1RVX preprocessing. Use --cohort primary for every model in the common comparison; no virtual attempts are invented for unsupported inputs.')
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Generation requires CUDA; evaluate and validate can run on CPU')
    from tasks import read_mol
    for task in tasks:
        core = read_mol(task['core_path'])
        if core.GetNumAtoms() != task['fixed_heavy_atoms']:
            raise ValueError(f"Core atom count changed: {task['case_id']}")
        if not 0 <= task['anchor_in_core'] < core.GetNumAtoms():
            raise ValueError(f"Invalid core anchor: {task['case_id']}")
    frozen = dict(format_version='github_release_v1', model=args.model, run=args.run,
        attempts=args.attempts, batch_size=args.batch_size, seed=args.seed,
        seed_strategy=selected['seed_strategy'], smoke=args.smoke, cohort=args.cohort,
        case_ids=[t['case_id'] for t in tasks], tasks_sha256=sha(args.tasks),
        inputs_sha256=input_hashes(tasks), source_sha256=source_hashes(),
        checkpoint_sha256=sha(ROOT/selected['checkpoint']),
        inference_keys=list(INFERENCE_KEYS), failed_and_duplicate_attempts_retained=True)
    if frozen['checkpoint_sha256'] != selected['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the recorded official checkpoint')
    dest = ROOT/'results'/args.run/args.model
    if (dest/'manifest.json').exists():
        previous = json.loads((dest/'manifest.json').read_text())
        if previous != frozen:
            raise ValueError('Run inputs, code or budget changed; choose a new run name')
    from assets import main as verify_assets
    from setup_sources import main as verify_sources
    verify_sources(['--model', args.model, '--verify'])
    verify_assets(['verify', '--model', args.model])
    dest.mkdir(parents=True, exist_ok=True)
    dump(dest/'manifest.json', frozen)
    dump(dest/'tasks.json', tasks)
    snapshots = dest/'source'
    for relative in frozen['source_sha256']:
        path = snapshots/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CODE_ROOT/relative, path)
    started = time.perf_counter()
    generate, adapter_config = getattr(importlib.import_module(selected['module']), selected['factory'])(args)
    dump(dest/'config.json', dict(**adapter_config, torch=torch.__version__,
         device=torch.cuda.get_device_name(0), loaded_seconds=time.perf_counter()-started))
    # MolJO enables gradients for its native critic internally; parameters stay frozen.
    with torch.no_grad():
        run_cases(generate, tasks, dest, args.model, args.attempts, args.batch_size,
                  args.seed, selected['seed_strategy'])
    dump(dest/'generation_complete.json', dict(model=args.model, cases=len(tasks),
         attempts=args.attempts*len(tasks), finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
