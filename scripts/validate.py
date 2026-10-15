"""CPU checks for prepared inputs and the main scientific guardrails."""
import argparse
import json
import subprocess
import sys
from benchmark_io import ROOT, CODE_ROOT, load_tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', default=str(ROOT/'data/benchmark_tasks.json'))
    parser.add_argument('--unit-tests', action='store_true')
    args = parser.parse_args(argv)
    if args.unit_tests:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(CODE_ROOT/'tests'), '-v'],
                       cwd=CODE_ROOT, check=True)
    from rdkit import Chem
    from tasks import audit, read_mol
    tasks = load_tasks(args.tasks)
    checked = []
    for task in tasks:
        ref = read_mol(task['reference_path'])
        positive = audit(ref, task)
        if not positive['strict_success']:
            raise ValueError(f'Reference control failed: {positive}')
        moved = Chem.Mol(ref)
        conf = moved.GetConformer()
        for index in task['fixed_reference_indices']:
            p = conf.GetAtomPosition(index)
            conf.SetAtomPosition(index, (p.x+1., p.y, p.z))
        if audit(moved, task)['strict_success']:
            raise ValueError('Moved-core negative control passed')
        wrong = dict(task, anchor_in_core=(task['anchor_in_core']+1) % task['fixed_heavy_atoms'])
        if audit(ref, wrong).get('single_site'):
            raise ValueError('Wrong-anchor negative control passed')
        checked.append(task['case_id'])
    print(json.dumps(dict(reference_controls=len(checked), moved_core_rejected=len(checked),
                         wrong_anchor_rejected=len(checked), passed=True)))
