#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
run_root="ai-society/runs/deliberation-s12-and-s06-extended-large-20260522"
log_dir="$run_root/logs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$log_dir"
echo "[$(date -Is)] deliberation s12 plus s06 extended large start log_dir=$log_dir"
PYTHONPATH=. uv run python ai-society/run_long_model_society_matrix.py \
  --config-list ai-society/configs/deliberation-s12-and-s06-extended-large-20260522/config-list.txt \
  --log-dir "$log_dir" \
  --continue-on-failure \
  > "$log_dir/controller.stdout.log" 2>&1
