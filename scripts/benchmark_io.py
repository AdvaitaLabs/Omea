"""Paths, manifests and lossless molecule archives shared by the benchmark."""
import hashlib
import json
import os
import re
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('RGROUP_BENCHMARK_ROOT', str(CODE_ROOT))).expanduser().resolve()
INFERENCE_KEYS = ('case_id', 'protein_path', 'core_path', 'anchor_in_core',
                  'rgroup_heavy_atoms', 'fixed_heavy_atoms')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Expected one directory name, got {value!r}')
    return value


def methods():
    return {r['id']: r for r in json.loads((CODE_ROOT/'configs/methods.json').read_text())}


def load_tasks(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('Task manifest must be a nonempty list')
    if len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate case IDs')
    for row in rows:
        safe_name(row['case_id'])
        for key in ('protein_path', 'core_path', 'reference_path'):
            p = Path(row[key]).expanduser()
            row[key] = str(p if p.is_absolute() else ROOT/p)
    return rows


def source_hashes():
    paths = [CODE_ROOT/'benchmark.py', *sorted((CODE_ROOT/'scripts').glob('*.py')),
             *sorted((CODE_ROOT/'configs').glob('*.json')), *sorted((CODE_ROOT/'patches').glob('*.patch'))]
    return {str(p.relative_to(CODE_ROOT)): sha(p) for p in paths}


def input_hashes(tasks):
    return {str(p): sha(p) for p in sorted({Path(r[k]) for r in tasks
            for k in ('protein_path', 'core_path', 'reference_path')})}


def save_molecules(folder, molecules):
    """Write native graphs before SDF export; exporting a copy cannot mutate them."""
    from rdkit import Chem
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    with Chem.SDWriter(str(folder/'raw.sdf')) as writer:
        writer.SetKekulize(False)
        for index, molecule in molecules:
            copied = Chem.Mol(molecule)
            copied.SetIntProp('attempt', index)
            path = folder/f'{index:04d}.rdmol'
            path.write_bytes(copied.ToBinary(Chem.PropertyPickleOptions.AllProps |
                                             Chem.PropertyPickleOptions.CoordsAsDouble))
            writer.write(Chem.Mol(copied))
            records.append(dict(attempt=index, file=path.name, sha256=sha(path)))
    dump(folder/'molecules.json', records)
    return records


def read_saved(folder, index):
    from rdkit import Chem
    return Chem.Mol((Path(folder)/f'{index:04d}.rdmol').read_bytes())
