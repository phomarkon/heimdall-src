#!/usr/bin/env bash
set -euo pipefail

cd /home/ucloud/heimdall
export PYTHONPATH=.

ROOT="ai-society/configs/chooser-det-llm-20260522"
RUN_ROOT="ai-society/runs/chooser-det-llm-20260522"
UPSTREAM_SESSION="heimdall-delib-s06-large-20260521"
UPSTREAM_RUN_ROOT="ai-society/runs/deliberation-s06-scenario-large-20260521"
UPSTREAM_EXPECTED=3
TARGET_SESSION="heimdall-chooser-det-llm"
TARGET_STAGE="chooser-det-llm-20260522"
TARGET_EXPECTED=45

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launch.lock"
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another chooser-det-llm launcher owns the lock; exiting"
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

latest_upstream_results() {
  find "$UPSTREAM_RUN_ROOT"/logs -path '*/results.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

wait_for_upstream() {
  while true; do
    local results
    results="$(latest_upstream_results || true)"
    local status="missing-results"
    if [ -n "$results" ] && [ -f "$results" ]; then
      status="$(stage_status "$results" "$UPSTREAM_EXPECTED")"
      case "$status" in
        failed)
          echo "[$(timestamp)] upstream deliberation matrix failed; refusing chooser-det-llm launch." >&2
          cat "$results" >&2
          exit 1
          ;;
        complete)
          echo "[$(timestamp)] upstream deliberation matrix complete with $UPSTREAM_EXPECTED successful rows"
          break
          ;;
      esac
    fi
    if ! tmux has-session -t "=$UPSTREAM_SESSION" 2>/dev/null && ! pgrep -af "run_long_model_society_matrix.py --config-list ai-society/configs/deliberation-s06-scenario-large-20260521/config-list.txt" >/dev/null; then
      echo "[$(timestamp)] upstream is not running and did not complete cleanly ($status)." >&2
      [ -n "$results" ] && [ -f "$results" ] && cat "$results" >&2 || true
      exit 1
    fi
    echo "[$(timestamp)] waiting for upstream deliberation matrix ($status)"
    sleep 300
  done
}

validate_configs() {
  uv run python tools/experiments/generate_chooser_det_llm_20260522.py --check-only >/dev/null
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
  pgrep -af "run_long_model_society_matrix.py --config-list $ROOT/all.txt" >/dev/null
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
  printf '%s\n' "$log_dir" > "$ROOT/latest-smoke-log-dir.txt"
  echo "[$(timestamp)] smoke start $TARGET_STAGE log_dir=$log_dir"
  uv run python ai-society/run_long_model_society_matrix.py     --config-list "$ROOT/smoke.txt"     --log-dir "$log_dir"     --continue-on-failure     --skip-vllm-restart     > "$log_dir/controller.stdout.log" 2>&1
  local status
  status="$(stage_status "$log_dir/results.json" 3)"
  if [ "$status" != "complete" ]; then
    echo "[$(timestamp)] smoke failed or incomplete: $status" >&2
    cat "$log_dir/results.json" >&2 || true
    exit 1
  fi
  echo "[$(timestamp)] smoke complete $TARGET_STAGE"
}

launch_full_stage() {
  prepare_target_session || return 0
  local log_dir="$RUN_ROOT/logs-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s\n' "$log_dir" > "$ROOT/latest-log-dir.txt"
  tmux new-session -d -s "$TARGET_SESSION" "
cd /home/ucloud/heimdall &&
export PYTHONPATH=. &&
uv run python ai-society/run_long_model_society_matrix.py   --config-list '$ROOT/all.txt'   --log-dir '$log_dir'   --continue-on-failure   --skip-vllm-restart   > '$log_dir/controller.stdout.log' 2>&1
"
  verify_launch "$log_dir"
  echo "[$(timestamp)] launched $TARGET_SESSION for $TARGET_STAGE log_dir=$log_dir"
}

verify_launch() {
  local log_dir="$1"
  for _ in $(seq 1 12); do
    if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null && matching_runner_active && grep -q "1/$TARGET_EXPECTED start" "$log_dir/controller.stdout.log" 2>/dev/null; then
      echo "[$(timestamp)] verified $TARGET_SESSION started $TARGET_STAGE"
      return 0
    fi
    sleep 5
  done
  echo "Launch verification failed for $TARGET_SESSION." >&2
  tmux has-session -t "=$TARGET_SESSION" 2>/dev/null || echo "missing tmux session" >&2
  matching_runner_active || echo "missing matching runner process" >&2
  tail -80 "$log_dir/controller.stdout.log" >&2 || true
  exit 1
}

wait_for_upstream
validate_configs
run_smoke_stage
launch_full_stage
