#!/usr/bin/env bash
set -euo pipefail

cd /home/ucloud/heimdall
export PYTHONPATH=.

ROOT="ai-society/configs/thesis-s06-equal-ablation-20260519"
RUN_ROOT="ai-society/runs/thesis-s06-equal-ablation-20260519"
STAGE="thesis-s06-equal-ablation"
BASE_URL="http://127.0.0.1:8000/v1"

validate_list() {
  local list_path="$1"
  while read -r cfg; do
    [ -z "$cfg" ] && continue
    uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
  done < "$list_path"
}

run_stage() {
  local config_list="$1"
  local log_dir="$2"
  uv run python ai-society/run_market_intelligence_stage.py \
    --stage "$STAGE" \
    --gpu gpu0 \
    --base-url "$BASE_URL" \
    --config-list "$config_list" \
    --log-dir "$log_dir" \
    > "$log_dir/sequential.stdout.log" 2>&1
}

uv run python tools/experiments/generate_thesis_s06_equal_ablation.py --check-only
validate_list "$ROOT/smoke.txt"
validate_list "$ROOT/all.txt"

smoke_log_dir="$RUN_ROOT/logs-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$smoke_log_dir"
printf '%s\n' "$smoke_log_dir" > "$ROOT/latest-smoke-log-dir.txt"
run_stage "$ROOT/smoke.txt" "$smoke_log_dir"

full_log_dir="$RUN_ROOT/logs-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$full_log_dir"
printf '%s\n' "$full_log_dir" > "$ROOT/latest-log-dir.txt"
run_stage "$ROOT/all.txt" "$full_log_dir"
