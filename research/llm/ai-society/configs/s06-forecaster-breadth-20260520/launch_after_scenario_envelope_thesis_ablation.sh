#!/usr/bin/env bash
set -euo pipefail

cd /home/ucloud/heimdall
export PYTHONPATH=.

BASE_URL="http://127.0.0.1:8000/v1"
UPSTREAM_ROOT="ai-society/configs/scenario-envelope-thesis-ablation-20260520"
UPSTREAM_EXPECTED=30

ROOT="ai-society/configs/s06-forecaster-breadth-20260520"
RUN_ROOT="ai-society/runs/s06-forecaster-breadth-20260520"
TARGET_SESSION="heimdall-s06-fb"
TARGET_STAGE="s06-forecaster-breadth"
TARGET_EXPECTED=36

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launch.lock"
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another s06-forecaster-breadth launcher owns the lock; exiting"
  exit 0
fi

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

stage_status() {
  local results="$1"
  local expected="$2"
  RESULTS="$results" EXPECTED="$expected" uv run python - <<'PY_STATUS'
import json
import os
from pathlib import Path

path = Path(os.environ["RESULTS"])
expected = int(os.environ["EXPECTED"])
rows = json.loads(path.read_text()) if path.exists() else []
failed = [row for row in rows if row.get("ok") is False]
ok = [row for row in rows if row.get("ok") is True]
if failed:
    print("failed")
elif len(ok) >= expected:
    print("complete")
else:
    print(f"running:{len(ok)}")
PY_STATUS
}

wait_for_upstream() {
  while true; do
    local status="missing-log-dir"
    local results=""
    if [ -f "$UPSTREAM_ROOT/latest-log-dir.txt" ]; then
      local log_dir
      log_dir="$(cat "$UPSTREAM_ROOT/latest-log-dir.txt")"
      results="$log_dir/gpu0-results.json"
      status="missing-results"
      if [ -f "$results" ]; then
        status="$(stage_status "$results" "$UPSTREAM_EXPECTED")"
        case "$status" in
          failed)
            echo "[$(timestamp)] upstream scenario-envelope-thesis-ablation failed; refusing s06 forecaster launch." >&2
            cat "$results" >&2
            exit 1
            ;;
          complete)
            echo "[$(timestamp)] upstream scenario-envelope-thesis-ablation complete with $UPSTREAM_EXPECTED successful rows"
            break
            ;;
        esac
      fi
    fi
    echo "[$(timestamp)] waiting for upstream scenario-envelope-thesis-ablation ($status)"
    sleep 300
  done
}

validate_configs() {
  uv run python tools/experiments/generate_s06_forecaster_breadth.py >/dev/null
  uv run python tools/experiments/generate_s06_forecaster_breadth.py --check-only >/dev/null
  while read -r cfg; do
    [ -z "$cfg" ] && continue
    uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
  done < "$ROOT/smoke.txt"
  while read -r cfg; do
    [ -z "$cfg" ] && continue
    uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
  done < "$ROOT/all.txt"
}

matching_runner_active() {
  pgrep -af "run_market_intelligence_stage.py --stage $TARGET_STAGE .*--log-dir $RUN_ROOT/logs-" >/dev/null
}

prepare_target_session() {
  if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
    if matching_runner_active; then
      echo "tmux session $TARGET_SESSION already runs $TARGET_STAGE; not launching duplicate."
      return 1
    fi
    echo "tmux session $TARGET_SESSION exists but no matching runner is active; removing stale session."
    tmux kill-session -t "=$TARGET_SESSION"
  fi
  return 0
}

run_smoke_stage() {
  local log_dir="$RUN_ROOT/logs-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s
' "$log_dir" > "$ROOT/latest-smoke-log-dir.txt"
  echo "[$(timestamp)] smoke start $TARGET_STAGE log_dir=$log_dir"
  uv run python ai-society/run_market_intelligence_stage.py \
    --stage "$TARGET_STAGE-smoke" \
    --gpu gpu0 \
    --base-url "$BASE_URL" \
    --config-list "$ROOT/smoke.txt" \
    --log-dir "$log_dir" \
    > "$log_dir/sequential.stdout.log" 2>&1
  echo "[$(timestamp)] smoke complete $TARGET_STAGE"
}

launch_full_stage() {
  prepare_target_session || return 0
  local log_dir="$RUN_ROOT/logs-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s
' "$log_dir" > "$ROOT/latest-log-dir.txt"
  tmux new-session -d -s "$TARGET_SESSION" "
cd /home/ucloud/heimdall &&
export PYTHONPATH=. &&
uv run python ai-society/run_market_intelligence_stage.py   --stage '$TARGET_STAGE'   --gpu gpu0   --base-url '$BASE_URL'   --config-list '$ROOT/all.txt'   --log-dir '$log_dir'   > '$log_dir/sequential.stdout.log' 2>&1
"
  verify_launch "$log_dir"
  wait_for_stage_results "$log_dir/gpu0-results.json"
}

verify_launch() {
  local log_dir="$1"
  for _ in $(seq 1 12); do
    if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null && matching_runner_active && grep -q "1/$TARGET_EXPECTED start" "$log_dir/sequential.stdout.log" 2>/dev/null; then
      echo "[$(timestamp)] verified $TARGET_SESSION started $TARGET_STAGE"
      return 0
    fi
    sleep 5
  done
  echo "Launch verification failed for $TARGET_SESSION." >&2
  tmux has-session -t "=$TARGET_SESSION" 2>/dev/null || echo "missing tmux session" >&2
  matching_runner_active || echo "missing matching runner process" >&2
  tail -80 "$log_dir/sequential.stdout.log" >&2 || true
  exit 1
}

wait_for_stage_results() {
  local results="$1"
  while true; do
    local status="missing-results"
    if [ -f "$results" ]; then
      status="$(stage_status "$results" "$TARGET_EXPECTED")"
      case "$status" in
        failed)
          echo "[$(timestamp)] s06-forecaster-breadth failed." >&2
          cat "$results" >&2
          exit 1
          ;;
        complete)
          echo "[$(timestamp)] s06-forecaster-breadth complete with $TARGET_EXPECTED successful rows"
          uv run python tools/evaluation/compare_s06_forecaster_breadth.py >/dev/null
          return 0
          ;;
      esac
    fi
    if ! tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
      echo "[$(timestamp)] $TARGET_SESSION is gone but $results does not show $TARGET_EXPECTED successful rows." >&2
      [ -f "$results" ] && cat "$results" >&2 || true
      exit 1
    fi
    echo "[$(timestamp)] waiting for $TARGET_SESSION ($status)"
    sleep 300
  done
}

wait_for_upstream
validate_configs
run_smoke_stage
launch_full_stage

echo "[$(timestamp)] s06-forecaster-breadth matrix complete"
