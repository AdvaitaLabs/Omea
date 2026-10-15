"""Rebuild the published cohort from public structures, or existing MOAD caches."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from benchmark_io import ROOT, CODE_ROOT, dump, sha


def freeze():
    from tasks import audit, read_mol
    pool = {r['case_id']: r for r in json.loads((ROOT/'data/moad_tasks.json').read_text())}
    frozen = json.loads((CODE_ROOT/'cohorts/single_site_45.json').read_text())
    selected = []
    for entry in frozen:
        if entry['case_id'] not in pool:
            raise ValueError(f"Frozen case missing after preparation: {entry['case_id']}")
        row = dict(pool[entry['case_id']])
        for key, value in entry.items():
            if key in row and row[key] != value:
                raise ValueError(f"Frozen task differs: {entry['case_id']}, {key}")
        row.update(entry)
        checked = audit(read_mol(row['reference_path']), row)
        if not checked['strict_success']:
            raise ValueError(f"Reference control failed: {entry['case_id']}: {checked}")
        for key in ('protein_path', 'core_path', 'reference_path', 'anchor_smiles_path'):
            row[key] = str(Path(row[key]).relative_to(ROOT))
        selected.append(row)
    dump(ROOT/'data/benchmark_tasks.json', selected)
    dump(ROOT/'data/cohort_manifest.json', dict(cases=len(selected), primary_cases=sum(r['primary'] for r in selected),
         primary_target_groups=len({r['target_group'] for r in selected if r['primary']}),
         frozen_cohort_sha256=sha(CODE_ROOT/'cohorts/single_site_45.json'),
         task_manifest_sha256=sha(ROOT/'data/benchmark_tasks.json'),
         exact_paper_replication=False, clean_held_out_claim=False))
    print(json.dumps(dict(cases=len(selected), primary_cases=sum(r['primary'] for r in selected),
                         tasks=str(ROOT/'data/benchmark_tasks.json'))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-cache', action='store_true', help='Use previously downloaded MOAD and CCD files')
    parser.add_argument('--freeze-only', action='store_true', help='Use an existing data/moad_tasks.json')
    args = parser.parse_args(argv)
    (ROOT/'data').mkdir(parents=True, exist_ok=True)
    (ROOT/'results').mkdir(parents=True, exist_ok=True)
    if args.from_cache and not (ROOT/'data/official_moad_test_ids.json').exists():
        dump(ROOT/'data/official_moad_test_ids.json', (CODE_ROOT/'cohorts/moad_test.txt').read_text().strip().split(','))
    if not args.freeze_only:
        if not args.from_cache:
            subprocess.run([sys.executable, str(CODE_ROOT/'scripts/fetch_moad.py')], check=True)
            from audit_ccd import fetch
            from concurrent.futures import ThreadPoolExecutor
            (ROOT/'data/ccd').mkdir(exist_ok=True)
            ids = json.loads((ROOT/'data/official_moad_test_ids.json').read_text())
            codes = sorted({r.split('_')[1].split(':')[0] for r in ids})
            with ThreadPoolExecutor(max_workers=6) as pool:
                reports = list(pool.map(fetch, codes))
            dump(ROOT/'results/ccd_download_manifest.json', reports)
            failures = [r for r in reports if r['status'] != 'downloaded']
            if failures:
                raise RuntimeError(f'{len(failures)} CCD downloads failed; inspect the manifest and retry')
        import prepare_moad
        prepare_moad.main()
    freeze()
