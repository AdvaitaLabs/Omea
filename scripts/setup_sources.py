"""Check out the recorded upstream commits and apply the recorded local patches."""
import argparse
import json
import subprocess
from pathlib import Path
from benchmark_io import ROOT, CODE_ROOT, sha


def git(directory, *args):
    return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Install only this model\'s source repository')
    parser.add_argument('--verify', action='store_true', help='Verify existing checkouts without downloading')
    parser.add_argument('--cache', type=Path, help='Optional local directory containing upstream git repositories')
    args = parser.parse_args(argv)
    from benchmark_io import methods
    selected = methods()[args.model]['source_repository'] if args.model else None
    repositories = json.loads((CODE_ROOT/'configs/sources.json').read_text())
    for row in repositories:
        if selected and row['name'] != selected:
            continue
        directory = ROOT/'vendor'/row['name']
        if not (directory/'.git').exists():
            if args.verify:
                raise FileNotFoundError(f'Missing git checkout: {directory}')
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(f'Refusing to replace existing non-git files: {directory}')
            directory.parent.mkdir(parents=True, exist_ok=True)
            if args.cache and (args.cache/row['name']/'.git').exists():
                subprocess.run(['git', 'clone', '--no-hardlinks', str(args.cache/row['name']), str(directory)], check=True)
            else:
                directory.mkdir(exist_ok=True)
                subprocess.run(['git', 'init', str(directory)], check=True)
                subprocess.run(['git', '-C', str(directory), 'remote', 'add', 'origin', row['url']], check=True)
                subprocess.run(['git', '-C', str(directory), 'fetch', '--depth', '1', 'origin', row['commit']], check=True)
            subprocess.run(['git', '-C', str(directory), 'checkout', '--detach', row['commit']], check=True)
        if git(directory, 'rev-parse', 'HEAD') != row['commit']:
            raise ValueError(f'Upstream revision changed: {row["name"]}')
        patch_name = row.get('patch')
        if patch_name:
            patch = CODE_ROOT/patch_name
            if sha(patch) != row['patch_sha256']:
                raise ValueError('Recorded patch changed')
            already = subprocess.run(['git', '-C', str(directory), 'apply', '--reverse', '--check', str(patch)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if not already:
                if args.verify:
                    raise ValueError(f'Recorded patch is missing: {row["name"]}')
                subprocess.run(['git', '-C', str(directory), 'apply', '--check', str(patch)], check=True)
                subprocess.run(['git', '-C', str(directory), 'apply', str(patch)], check=True)
        actual = subprocess.check_output(['git', '-C', str(directory), 'diff', '--binary', '--no-ext-diff'])
        expected = (CODE_ROOT/patch_name).read_bytes() if patch_name else b''
        if actual != expected:
            raise ValueError(f'Additional unrecorded upstream edits: {row["name"]}')
        print(json.dumps(dict(repository=row['name'], commit=row['commit'], verified=True)), flush=True)
