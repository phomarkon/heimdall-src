#!/usr/bin/env bash
set -euo pipefail
cd /home/ucloud/heimdall
export PYTHONPATH='.'
run_root='ai-society/runs/deliberation-s06-scenario-large-20260521'
log_dir="$run_root/logs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$log_dir"
echo "[$(date -Is)] deliberation s06 scenario large start log_dir=$log_dir"
uv run python ai-society/run_long_model_society_matrix.py \
  --config-list ai-society/configs/deliberation-s06-scenario-large-20260521/config-list.txt \
  --log-dir "$log_dir" \
  --continue-on-failure \
  > "$log_dir/controller.stdout.log" 2>&1
echo "[$(date -Is)] deliberation s06 scenario large complete log_dir=$log_dir"
