"""Score references and strict outputs in their unchanged generated poses."""
import argparse
import concurrent.futures
import importlib.metadata
import json
import shutil
from pathlib import Path
from benchmark_io import ROOT, CODE_ROOT, dump, read_saved, safe_name, sha
from metrics import geometry_result


def score_cached(molecule, task, out, judge):
    from rdkit import Chem
    from tasks import audit
    from scoring import score_one
    checked = audit(molecule, task)
    if not checked['strict_success']:
        raise ValueError(f'Archived strict molecule does not reproduce its audit: {checked}')
    fingerprint = dict(judge=judge, protein_sha256=sha(task['protein_path']),
        molecule_sha256=__import__('hashlib').sha256(molecule.ToBinary(
            Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble)).hexdigest())
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
        raise ValueError(f'Cached judge input changed: {out}; use a new run')
    if (out/'score.json').exists() and not manifest.exists():
        raise ValueError(f'Unverified scoring cache: {out}')
    dump(manifest, fingerprint)
    result = dict(score_one(molecule, task['protein_path'], out))
    passed, missing = geometry_result(result)
    result.update(posebusters_pass=passed, posebusters_complete=not missing,
                  missing_checks=missing)
    dump(out/'score.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gnina', help='GNINA 1.1 executable; defaults to GNINA_BIN or tools/gnina')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    safe_name(args.run); safe_name(args.model)
    import scoring
    if args.gnina:
        scoring.GNINA = Path(args.gnina).expanduser().resolve()
    elif not scoring.GNINA.is_file():
        found = shutil.which('gnina')
        if found:
            scoring.GNINA = Path(found)
    if not scoring.GNINA.is_file():
        raise FileNotFoundError('GNINA 1.1 is missing; pass --gnina PATH or set GNINA_BIN')
    from tasks import read_mol
    dest = ROOT/'results'/args.run/args.model
    frozen = json.loads((dest/'manifest.json').read_text())
    if not (dest/'generation_complete.json').exists():
        raise ValueError('Generation is not complete; resume generation before scoring')
    tasks = json.loads((dest/'tasks.json').read_text())
    judge = dict(mode='fixed_pose_score_only', gnina_sha256=sha(scoring.GNINA),
        rdkit=importlib.metadata.version('rdkit'), posebusters=importlib.metadata.version('posebusters'),
        scoring_source_sha256=sha(CODE_ROOT/'scripts/scoring.py'),
        audit_source_sha256=sha(CODE_ROOT/'scripts/tasks.py'),
        metrics_source_sha256=sha(CODE_ROOT/'scripts/metrics.py'),
        geometry_schema_sha256=sha(CODE_ROOT/'configs/geometry_schema.json'))
    jobs, expected = [], 0
    for task in tasks:
        folder = dest/task['case_id']
        attempts = json.loads((folder/'attempts.json').read_text())
        if len(attempts) != frozen['attempts'] or {r['attempt'] for r in attempts} != set(range(frozen['attempts'])):
            raise ValueError('Incomplete attempt ledger')
        archived = {r['attempt']: r for r in json.loads((folder/'molecules.json').read_text())}
        jobs.append(('reference', task, None, read_mol(task['reference_path']),
                     ROOT/'results'/args.run/'judges/reference'/task['case_id']))
        for row in attempts:
            if not row['strict_success']:
                continue
            index = row['attempt']
            entry = archived[index]
            if sha(folder/entry['file']) != entry['sha256']:
                raise ValueError('Archived molecule changed')
            molecule = read_saved(folder, index)
            from tasks import audit
            checked = audit(molecule, task)
            if not checked['strict_success'] or checked['smiles'] != row['smiles']:
                raise ValueError(f"Archived identity changed: {task['case_id']}, {index}")
            jobs.append((args.model, task, index, molecule,
                         ROOT/'results'/args.run/'judges'/args.model/task['case_id']/f'{index:04d}'))
            expected += 1
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_cached, mol, task, out, judge): (model, task, index)
                   for model, task, index, mol, out in jobs}
        for future in concurrent.futures.as_completed(futures):
            model, task, index = futures[future]
            results.append(dict(model=model, case_id=task['case_id'], attempt=index, **future.result()))
            if len(results) % 25 == 0 or len(results) == len(jobs):
                print(json.dumps(dict(model=args.model, scored_jobs=len(results), total_jobs=len(jobs))), flush=True)
    references = {r['case_id']: r for r in results if r['model'] == 'reference'}
    if any(references[t['case_id']]['status'] != 'scored' for t in tasks):
        dump(dest/'reference_scoring_failure.json', references)
        raise RuntimeError('Reference scoring failed; comparison was not produced')
    for row in results:
        ref = references[row['case_id']]
        if row['model'] != 'reference' and row['status'] == 'scored':
            row['delta_vina_kcal_mol'] = row['vina_kcal_mol']-ref['vina_kcal_mol']
            row['delta_cnn_affinity_pK'] = row['cnn_affinity_pK']-ref['cnn_affinity_pK']
    results.sort(key=lambda r: (r['model'], r['case_id'], -1 if r['attempt'] is None else r['attempt']))
    dump(dest/'judge_results.json', dict(judge=judge, strict_candidate_jobs=expected,
         rows=results, interpretation='Predicted proxy changes; neither metric is physical DDG'))
    dump(dest/'scoring_complete.json', dict(jobs=len(results), strict_candidate_jobs=expected,
         failed_candidates=sum(r['model'] != 'reference' and r['status'] != 'scored' for r in results)))
