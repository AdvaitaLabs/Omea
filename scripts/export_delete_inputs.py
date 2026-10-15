"""Export retained-core coordinates for DELETE's isolated Python 3.7 surface environment."""
import argparse
from benchmark_io import ROOT, dump, load_tasks, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', default=str(ROOT/'data/benchmark_tasks.json'))
    parser.add_argument('--out', default=str(ROOT/'data/delete_surface_inputs.json'))
    args = parser.parse_args()
    from tasks import read_mol
    rows = []
    for task in load_tasks(args.tasks):
        core = read_mol(task['core_path'])
        rows.append(dict(case_id=task['case_id'], protein_path=task['protein_path'],
             core_xyz=core.GetConformer().GetPositions().tolist(),
             core_sha256=sha(task['core_path']), protein_sha256=sha(task['protein_path'])))
    dump(args.out, rows)


if __name__ == '__main__':
    main()
