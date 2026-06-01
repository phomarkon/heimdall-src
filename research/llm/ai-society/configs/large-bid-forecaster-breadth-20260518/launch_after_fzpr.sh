#!/usr/bin/env bash
set -euo pipefail
cd /home/ucloud/heimdall
export PYTHONPATH=.

CURRENT_LOG_DIR="$(cat ai-society/configs/forecaster-zoo-priority-realcontrol-20260518/latest-log-dir.txt)"
CURRENT_RESULTS="$CURRENT_LOG_DIR/gpu0-results.json"
ROOT="ai-society/configs/large-bid-forecaster-breadth-20260518"

wait_for_current() {
  while true; do
    status="missing"
    if [ -f "$CURRENT_RESULTS" ]; then
      status="$(CURRENT_RESULTS="$CURRENT_RESULTS" uv run python - <<'PY_STATUS'
import json
import os
from pathlib import Path
path = Path(os.environ['CURRENT_RESULTS'])
rows = json.loads(path.read_text()) if path.exists() else []
failed = [r for r in rows if r.get('ok') is False]
ok = [r for r in rows if r.get('ok') is True]
if failed:
    print('failed')
elif len(ok) >= 12:
    print('complete')
else:
    print(f'running:{len(ok)}')
PY_STATUS
)"
      case "$status" in
        failed)
          echo "Current matrix has failures; refusing to launch next matrix." >&2
          cat "$CURRENT_RESULTS" >&2
          exit 1
          ;;
        complete)
          break
          ;;
      esac
    fi
    if ! tmux has-session -t heimdall-fzpr 2>/dev/null; then
      echo "heimdall-fzpr is gone but current matrix does not show 12 successful rows; refusing to launch." >&2
      cat "$CURRENT_RESULTS" >&2 || true
      exit 1
    fi
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for current matrix ($status)"
    sleep 300
  done
}

validate_configs() {
  while read -r cfg; do
    uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1
  done < "$ROOT/all.txt"
  while read -r cfg; do
    uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1
  done < "$ROOT/smoke.txt"
}

run_smoke() {
  while read -r cfg; do
    echo "[smoke] $cfg"
    uv run python -m heimdall_ai_society run --config "$cfg" || exit 1
  done < "$ROOT/smoke.txt"
}

launch_full() {
  if tmux has-session -t heimdall-lbfb 2>/dev/null; then
    echo "tmux session heimdall-lbfb already exists; refusing duplicate launch." >&2
    exit 1
  fi
  log_dir="ai-society/runs/large-bid-forecaster-breadth-20260518/logs-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s
' "$log_dir" > "$ROOT/latest-log-dir.txt"
  tmux new-session -d -s heimdall-lbfb "
cd /home/ucloud/heimdall &&
export PYTHONPATH=. &&
uv run python ai-society/run_market_intelligence_stage.py \
  --stage large-bid-forecaster-breadth \
  --gpu gpu0 \
  --base-url http://127.0.0.1:8000/v1 \
  --config-list '$ROOT/all.txt' \
  --log-dir '$log_dir' \
  > '$log_dir/sequential-2gpu.stdout.log' 2>&1
"
  echo "$log_dir"
}

wait_for_current
validate_configs
run_smoke
launch_full
