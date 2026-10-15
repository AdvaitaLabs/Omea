"""Build per-case and target-weighted tables from complete, matched runs."""
import argparse
import collections
import csv
import json
from benchmark_io import ROOT, dump, safe_name
from metrics import outcomes

COUNT_KEYS = ('attempts', 'chemical_valid', 'strict_success', 'valid_with_geometry',
              'valid_new', 'dual_improved', 'charge_matched_dual', 'unique_valid_new',
              'unique_dual_improved', 'score_failures', 'missing_geometry_checks')


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def case_summary(model, task, attempts, scores, reference):
    evaluated = []
    for row in attempts:
        score = scores.get((task['case_id'], row['attempt']), {})
        if row['strict_success'] and not score:
            raise ValueError(f"Missing strict score: {task['case_id']}, {row['attempt']}")
        evaluated.append((row, score, outcomes(row, score, reference)))
    new = [a for a, s, o in evaluated if o['valid_new']]
    dual = [a for a, s, o in evaluated if o['dual_improved']]
    return dict(model=model, case_id=task['case_id'], pdb_id=task['pdb_id'],
        stratum=task['stratum'], target_group=task['target_group'], primary=task['primary'],
        attempts=len(attempts), chemical_valid=sum(bool(a['valid']) for a in attempts),
        strict_success=sum(bool(a['strict_success']) for a in attempts),
        valid_with_geometry=sum(o['valid_with_geometry'] for a, s, o in evaluated),
        valid_new=len(new), dual_improved=len(dual),
        charge_matched_dual=sum(o['charge_matched_dual'] for a, s, o in evaluated),
        unique_valid_new=len({a['smiles'] for a in new}),
        unique_dual_improved=len({a['smiles'] for a in dual}),
        score_failures=sum(a['strict_success'] and s.get('status') != 'scored' for a, s, o in evaluated),
        missing_geometry_checks=sum(a['strict_success'] and s.get('status') == 'scored' and
                                    bool(o['missing_geometry_checks']) for a, s, o in evaluated))


def aggregate(model, rows):
    sums = {key: sum(r[key] for r in rows) for key in COUNT_KEYS}
    targets = collections.defaultdict(list)
    for row in rows:
        targets[row['target_group']].append(row)
    def target_mean(metric):
        return 100*sum(sum(r[metric]/r['attempts'] for r in group)/len(group)
                       for group in targets.values())/len(targets)
    return dict(model=model, cases=len(rows), target_groups=len(targets), **sums,
        valid_new_percent=100*sums['valid_new']/sums['attempts'],
        dual_improved_percent=100*sums['dual_improved']/sums['attempts'],
        target_equal_weight_valid_new_percent=target_mean('valid_new'),
        target_equal_weight_dual_improved_percent=target_mean('dual_improved'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--models', nargs='+', help='Default: all completed model directories')
    parser.add_argument('--smoke', action='store_true', help='Explicitly allow integration-test summaries')
    args = parser.parse_args(argv)
    safe_name(args.run)
    root = ROOT/'results'/args.run
    models = args.models or sorted(p.name for p in root.iterdir() if (p/'manifest.json').exists())
    if not models:
        raise ValueError('No generated model runs')
    cases, previous_signature, previous_judge = [], None, None
    for model in models:
        safe_name(model)
        folder = root/model
        frozen = json.loads((folder/'manifest.json').read_text())
        if frozen['smoke'] and not args.smoke:
            raise ValueError('Smoke outputs cannot enter the main benchmark summary')
        if not (folder/'scoring_complete.json').exists():
            raise ValueError(f'Model has not completed scoring: {model}')
        signature = {k: frozen[k] for k in ('attempts', 'seed', 'case_ids', 'tasks_sha256', 'inputs_sha256', 'source_sha256', 'smoke')}
        if previous_signature is not None and signature != previous_signature:
            raise ValueError('Models have different cohorts, inputs, seed, budgets or code snapshots')
        previous_signature = signature
        tasks = json.loads((folder/'tasks.json').read_text())
        judged = json.loads((folder/'judge_results.json').read_text())
        if previous_judge is not None and judged['judge'] != previous_judge:
            raise ValueError('Models were scored with different binaries, software or scoring definitions')
        previous_judge = judged['judge']
        references = {r['case_id']: r for r in judged['rows'] if r['model'] == 'reference'}
        scores = {(r['case_id'], r['attempt']): r for r in judged['rows'] if r['model'] == model}
        for task in tasks:
            attempts = json.loads((folder/task['case_id']/'attempts.json').read_text())
            if len(attempts) != frozen['attempts'] or {r['attempt'] for r in attempts} != set(range(frozen['attempts'])):
                raise ValueError('Incomplete or duplicated raw attempt ledger')
            cases.append(case_summary(model, task, attempts, scores, references[task['case_id']]))
    destination = root/('smoke_summary' if args.smoke else 'summary')
    destination.mkdir(exist_ok=True)
    write_csv(destination/'per_case.csv', cases)
    all_rows, primary_rows, target_rows = [], [], []
    for model in models:
        selected = [r for r in cases if r['model'] == model]
        all_rows.append(aggregate(model, selected))
        primary = [r for r in selected if r['primary']]
        if primary:
            primary_rows.append(aggregate(model, primary))
        for group in sorted({r['target_group'] for r in primary}):
            values = aggregate(model, [r for r in primary if r['target_group'] == group])
            target_rows.append(dict(target_group=group, **values))
    write_csv(destination/'comparison_all_cases.csv', all_rows)
    write_csv(destination/'comparison_primary_30.csv', primary_rows)
    write_csv(destination/'per_target_primary.csv', target_rows)
    dump(destination/'summary.json', dict(models=models, primary=primary_rows,
         all_cases=all_rows, smoke=args.smoke, unique_counts='sum of per-case unique counts, not globally deduplicated compounds',
         interpretation='Success percentages use every raw attempt. Scores are proxies, not experimental affinity or physical DDG.'))
    print(json.dumps(dict(directory=str(destination), models=len(models))))
