#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

BATCH_DIR="${1:?usage: monitor_profit_window_batch.sh <batch-dir>}"
LOG_FILE="$BATCH_DIR/monitor-10min.log"

RUNS=(
  diverse-action-apr13-0015-48-q32
  diverse-action-apr09-1830-48-q32
  diverse-action-apr02-0530-48-q32
  diverse-action-apr06-1300-48-q32
  diverse-action-apr26-1400-48-q32
  p2h-stresstest-apr03-1915-24-q32
  p2h-stresstest-apr17-0745-24-q32
  p2h-stresstest-apr25-1600-24-q32
  p2h-stresstest-apr22-1430-24-q32
  p2h-stresstest-apr01-0415-24-q32
)

while true; do
  {
    echo "=== $(date -u +%FT%TZ) ==="
    cat "$BATCH_DIR/status.json" 2>/dev/null || true
    for run_id in "${RUNS[@]}"; do
      trace_path="ai-society/runs/$run_id/traces.jsonl"
      summary_path="evaluations/$run_id/run_summary.json"
      if [[ -f "$trace_path" ]]; then
        printf 'run_rows %s ' "$run_id"
        wc -l < "$trace_path"
      fi
      if [[ -f "$summary_path" ]]; then
        uv run python - "$summary_path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    d = json.load(handle)
print(
    "eval "
    f"{d.get('run_id')} "
    f"profit={d.get('realized_profit_eur')} "
    f"filled={d.get('filled_count')} "
    f"mwh={d.get('cleared_mwh')} "
    f"side={d.get('side_precision')} "
    f"oracle={d.get('oracle_feasible_profit_eur')}"
)
PY
      fi
    done
  } | tee -a "$LOG_FILE"
  sleep 600
done
