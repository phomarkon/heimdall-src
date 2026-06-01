#!/usr/bin/env bash
set -euo pipefail

cd /home/ucloud/heimdall
export PYTHONPATH=.

ROOT="ai-society/configs/det-llm-critic-20260522"
RUN_ROOT="ai-society/runs/det-llm-critic-20260522"
UPSTREAM_RUN_ROOT="ai-society/runs/chooser-det-llm-20260522"
UPSTREAM_EXPECTED=45
TARGET_SESSION="heimdall-det-llm-critic-s06"
TARGET_EXPECTED=3

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launch.lock"
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another det-llm-critic launcher owns the lock; exiting"
  exit 0
fi

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

stage_status() {
  local results="$1"
  local expected="$2"
  RESULTS="$results" EXPECTED="$expected" uv run python - <<'PY_STATUS'
import json
import os
from pathlib import Path
rows = json.loads(Path(os.environ["RESULTS"]).read_text()) if Path(os.environ["RESULTS"]).exists() else []
failed = [row for row in rows if row.get("ok") is False]
ok = [row for row in rows if row.get("ok") is True]
if failed:
    print("failed")
elif len(ok) >= int(os.environ["EXPECTED"]):
    print("complete")
else:
    print(f"running:{len(ok)}")
PY_STATUS
}

latest_upstream_results() {
  find "$UPSTREAM_RUN_ROOT"/logs-* -name results.json -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
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
          echo "[$(timestamp)] upstream CDL matrix failed; refusing det-llm-critic launch." >&2
          cat "$results" >&2
          exit 1
          ;;
        complete)
          echo "[$(timestamp)] upstream CDL matrix complete with $UPSTREAM_EXPECTED successful rows"
          break
          ;;
      esac
    fi
    if ! tmux has-session -t "=heimdall-chooser-det-llm" 2>/dev/null && ! pgrep -af "run_long_model_society_matrix.py --config-list ai-society/configs/chooser-det-llm-20260522/all.txt" >/dev/null; then
      echo "[$(timestamp)] upstream CDL is not running and did not complete cleanly ($status)." >&2
      [ -n "$results" ] && [ -f "$results" ] && cat "$results" >&2 || true
      exit 1
    fi
    echo "[$(timestamp)] waiting for upstream CDL matrix ($status)"
    sleep 300
  done
}

validate_configs() {
  uv run python tools/experiments/generate_det_llm_critic_20260522.py --check-only >/dev/null
  while read -r cfg; do [ -z "$cfg" ] || uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null; done < "$ROOT/smoke.txt"
  while read -r cfg; do [ -z "$cfg" ] || uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null; done < "$ROOT/all.txt"
}

run_smoke_stage() {
  local log_dir="$RUN_ROOT/logs-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s\n' "$log_dir" > "$ROOT/latest-smoke-log-dir.txt"
  echo "[$(timestamp)] smoke start det-llm-critic-s06 log_dir=$log_dir"
  uv run python ai-society/run_long_model_society_matrix.py     --config-list "$ROOT/smoke.txt"     --log-dir "$log_dir"     --continue-on-failure     --skip-vllm-restart     > "$log_dir/controller.stdout.log" 2>&1
  local status
  status="$(stage_status "$log_dir/results.json" 3)"
  if [ "$status" != "complete" ]; then
    echo "[$(timestamp)] smoke failed or incomplete: $status" >&2
    cat "$log_dir/results.json" >&2 || true
    exit 1
  fi
}

launch_full_stage() {
  if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
    echo "tmux session $TARGET_SESSION already exists; not launching duplicate."
    return 0
  fi
  local log_dir="$RUN_ROOT/logs-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s\n' "$log_dir" > "$ROOT/latest-log-dir.txt"
  tmux new-session -d -s "$TARGET_SESSION" "
cd /home/ucloud/heimdall &&
export PYTHONPATH=. &&
uv run python ai-society/run_long_model_society_matrix.py   --config-list '$ROOT/all.txt'   --log-dir '$log_dir'   --continue-on-failure   --skip-vllm-restart   > '$log_dir/controller.stdout.log' 2>&1
"
  sleep 5
  if ! tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
    echo "[$(timestamp)] launch verification failed for $TARGET_SESSION" >&2
    tail -80 "$log_dir/controller.stdout.log" >&2 || true
    exit 1
  fi
  echo "[$(timestamp)] launched $TARGET_SESSION for 3 full S06 critic runs log_dir=$log_dir"
}

wait_for_upstream
validate_configs
run_smoke_stage
launch_full_stage
