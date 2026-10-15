#!/usr/bin/env python3
"""Public entry point. Model environments stay isolated in separate processes."""
import argparse
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMANDS = {
    'setup-sources': 'setup_sources',
    'assets': 'assets',
    'prepare': 'prepare',
    'generate': 'generate',
    'score': 'score',
    'summarize': 'summarize',
    'validate': 'validate',
}


def main():
    parser = argparse.ArgumentParser(description='Strict single-attachment R-group benchmark')
    parser.add_argument('command', choices=['models', *COMMANDS])
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    if args.command == 'models':
        if rest:
            parser.error('models takes no additional arguments')
        for row in json.loads((HERE/'configs/methods.json').read_text()):
            print(f"{row['id']:26s} {row['environment']:16s} {row['code_license']}")
        return
    sys.path.insert(0, str(HERE/'scripts'))
    importlib.import_module(COMMANDS[args.command]).main(rest)


if __name__ == '__main__':
    main()
