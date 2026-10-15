"""Download from official sources, or verify locally supplied checkpoints."""
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from benchmark_io import ROOT, CODE_ROOT, dump, methods, sha


def download(entry):
    path = ROOT/entry['path']
    if path.exists():
        if sha(path) != entry['sha256']:
            raise ValueError(f'Existing file checksum mismatch: {path}')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix+'.part')
    request = urllib.request.Request(entry['url'], headers={'User-Agent': 'rgroup-benchmark'})
    with urllib.request.urlopen(request, timeout=90) as response, part.open('wb') as out:
        if 'text/html' in response.headers.get('Content-Type', ''):
            raise ValueError('Provider returned an HTML page; obtain the file from its official page and run assets verify')
        shutil.copyfileobj(response, out, 4*1024*1024)
    if sha(part) != entry['sha256']:
        raise ValueError(f'Download checksum mismatch: {path}; incomplete .part retained')
    part.replace(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'download', 'verify'])
    parser.add_argument('--model', choices=list(methods()))
    args = parser.parse_args(argv)
    catalog = json.loads((CODE_ROOT/'configs/assets.json').read_text())
    if args.action == 'list':
        print(json.dumps(catalog[args.model] if args.model else catalog, indent=2))
        return
    if not args.model:
        parser.error('--model is required for download or verify')
    record = catalog[args.model]
    if args.action == 'download':
        installer = record.get('installer')
        for entry in record.get('downloads', []):
            archive = download(entry)
            if installer == 'pocketxmol_tar':
                target = ROOT/'checkpoints/pocketxmol'
                target.mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive) as source:
                    source.extractall(target, filter='data')
            elif installer == 'pfizer_zip':
                target = ROOT/'checkpoints/pfizer'
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive) as source:
                    # Only the selected cutable checkpoints are installed.
                    for required in record['required_files']:
                        relative = str(Path(required['path']).relative_to(target.relative_to(ROOT)))
                        member = source.getinfo(relative)
                        destination = target/relative
                        if not destination.resolve().is_relative_to(target.resolve()):
                            raise ValueError('Unsafe archive member')
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with source.open(member) as inp, destination.open('wb') as out:
                            shutil.copyfileobj(inp, out, 4*1024*1024)
        if installer == 'neat_ranges':
            (ROOT/'checkpoints').mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, str(CODE_ROOT/'scripts/download_neat_parallel.py')], check=True)
            from download_neat_weights import RemoteFile
            with zipfile.ZipFile(RemoteFile()) as source:
                required = next(r for r in record['required_files'] if 'bond_predictor/' in r['path'])
                relative = str(Path(required['path']).relative_to('checkpoints/neat_pocket'))
                destination = ROOT/required['path']
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(relative) as inp, destination.open('wb') as out:
                    shutil.copyfileobj(inp, out, 4*1024*1024)
    reports = []
    for entry in record['required_files']:
        path = ROOT/entry['path']
        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint or auxiliary file missing: {path}')
        digest = sha(path)
        if digest != entry['sha256']:
            raise ValueError(f'Checkpoint differs from benchmark weights: {path}')
        reports.append(dict(path=entry['path'], sha256=digest, bytes=path.stat().st_size))
    dump(ROOT/'results/assets'/f'{args.model}.json', dict(model=args.model, verified=reports))
    print(json.dumps(dict(model=args.model, verified_files=len(reports))))
