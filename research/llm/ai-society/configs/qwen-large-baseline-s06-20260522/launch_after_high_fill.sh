#!/usr/bin/env bash
set -euo pipefail
cd /home/ucloud/heimdall
export PYTHONPATH=.

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

while true; do
  summary=$(ls -td ai-society/runs/high-fill-llm-s06-20260522/logs-*/summary.json 2>/dev/null | head -n 1 || true)
  if [[ -n "$summary" ]]; then
    read -r completed failed < <(python - "$summary" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print(d.get("completed", 0), d.get("failed", 0))
PY
)
    if [[ "$completed" == "12" && "$failed" == "0" ]]; then
      log "upstream high-fill complete with $completed successful rows"
      break
    fi
    log "waiting for high-fill matrix (running:$completed failed:$failed)"
  else
    log "waiting for high-fill matrix (missing-results)"
  fi
  sleep 300
done

uv run python tools/experiments/generate_qwen_large_baseline_s06_20260522.py --check-only
while read -r cfg; do
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
done < ai-society/configs/qwen-large-baseline-s06-20260522/smoke.txt
while read -r cfg; do
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
done < ai-society/configs/qwen-large-baseline-s06-20260522/all.txt

log_dir=ai-society/runs/qwen-large-baseline-s06-20260522/logs-$(date -u +%Y%m%dT%H%M%SZ)
tmux new-session -d -s heimdall-qwen-large-baseline-s06 \
  "cd /home/ucloud/heimdall && PYTHONPATH=. uv run python ai-society/run_long_model_society_matrix.py --config-list ai-society/configs/qwen-large-baseline-s06-20260522/all.txt --log-dir $log_dir --continue-on-failure --health-timeout-seconds 1200 > $log_dir.controller.stdout.log 2>&1"
log "launched heimdall-qwen-large-baseline-s06 log_dir=$log_dir"
