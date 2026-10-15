#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_name="${1:-AutoFragDiff}"
run_name="${2:-benchmark_v1}"
# Activate the model's environment first. Checkpoints and inputs must be prepared.
python "$repo_dir/benchmark.py" generate --model "$model_name" --run "$run_name" --cohort primary
# Scoring can also be invoked later from the CPU evaluation environment.
python "$repo_dir/benchmark.py" score --model "$model_name" --run "$run_name"
python "$repo_dir/benchmark.py" summarize --run "$run_name" --models "$model_name"
