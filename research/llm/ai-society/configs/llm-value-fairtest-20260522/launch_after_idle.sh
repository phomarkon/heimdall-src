#!/usr/bin/env bash
# Chain the llm-value-fairtest matrix BEHIND the current GPU queue (central-supervisor-s06):
#   wait-for-idle -> validate -> 5x 2-tick smoke (reuses persistent vLLM)
#   -> hard gate (all smoke complete + LLM modes actually placed bids)
#   -> launch 30-run full matrix (core seeds 42,13 first, then 137).
# Fail-fast: broken smoke or all-abstain LLM stops before the ~6-7h full run.
# vLLM is reused (--skip-vllm-restart): the persistent Qwen3-32B servers on 8000/8001 stay up.
set -euo pipefail

cd /home/ucloud/heimdall
export PYTHONPATH=.

MATRIX="llm-value-fairtest-20260522"
ROOT="ai-society/configs/$MATRIX"
RUN_ROOT="ai-society/runs/$MATRIX"
GEN="tools/experiments/generate_llm_value_fairtest_20260522.py"
TARGET_SESSION="heimdall-fairtest"
SMOKE_EXPECTED=5
CHECKS_REQUIRED=2
POLL_SECONDS=300

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launch.lock"
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another fairtest launcher owns the lock; exiting"
  exit 0
fi

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

queue_busy() {
  pgrep -f "run_long_model_society_matrix.py" >/dev/null && return 0
  tmux has-session -t "=heimdall-central-supervisor-s06" 2>/dev/null && return 0
  return 1
}

wait_for_idle() {
  local idle=0
  while true; do
    if queue_busy; then
      idle=0
      echo "[$(timestamp)] queue busy (central-supervisor); waiting"
    else
      idle=$((idle + 1))
      echo "[$(timestamp)] queue idle check $idle/$CHECKS_REQUIRED"
      [ "$idle" -ge "$CHECKS_REQUIRED" ] && break
    fi
    sleep "$POLL_SECONDS"
  done
  echo "[$(timestamp)] queue idle confirmed; proceeding"
}

vllm_ready() {
  curl -fsS -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 || return 1
  curl -fsS -m 5 http://127.0.0.1:8001/v1/models >/dev/null 2>&1 || return 1
  return 0
}

validate() {
  uv run python "$GEN" --check-only >/dev/null
  while read -r cfg; do [ -z "$cfg" ] || uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null; done < "$ROOT/smoke.txt"
  while read -r cfg; do [ -z "$cfg" ] || uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null; done < "$ROOT/all.txt"
  echo "[$(timestamp)] configs validated"
}

run_smoke() {
  SMOKE_LOG_DIR="$RUN_ROOT/logs-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$SMOKE_LOG_DIR"
  printf '%s\n' "$SMOKE_LOG_DIR" > "$ROOT/latest-smoke-log-dir.txt"
  echo "[$(timestamp)] smoke start (reusing persistent vLLM) log_dir=$SMOKE_LOG_DIR"
  uv run python ai-society/run_long_model_society_matrix.py \
    --config-list "$ROOT/smoke.txt" \
    --log-dir "$SMOKE_LOG_DIR" \
    --continue-on-failure \
    --skip-vllm-restart \
    > "$SMOKE_LOG_DIR/controller.stdout.log" 2>&1 || true
}

smoke_completed() {
  python3 -c "import json; r=json.load(open('$SMOKE_LOG_DIR/results.json')); print(sum(1 for x in r if x.get('ok')))" 2>/dev/null || echo 0
}

# Gate: the LLM modes must actually place bids (not collapse to all-abstain). Reads the
# inline-eval run_summary.json (bid_action_count) rather than the unreliable autonomy metric.
llm_engaged() {
  python3 - <<'PY'
import glob, json
total = 0
for mode in ("selector", "comm", "memory", "cp12"):
    for path in glob.glob(f"evaluations/smoke-lvf-s06-{mode}-*/run_summary.json"):
        try:
            total += int(json.load(open(path)).get("bid_action_count") or 0)
        except Exception:
            pass
print("yes" if total > 0 else "no")
PY
}

launch_full() {
  if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
    echo "[$(timestamp)] $TARGET_SESSION already exists; not launching duplicate"
    return 0
  fi
  local log_dir="$RUN_ROOT/logs-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s\n' "$log_dir" > "$ROOT/latest-log-dir.txt"
  tmux new-session -d -s "$TARGET_SESSION" "
cd /home/ucloud/heimdall &&
export PYTHONPATH=. &&
uv run python ai-society/run_long_model_society_matrix.py --config-list '$ROOT/all.txt' --log-dir '$log_dir' --continue-on-failure --skip-vllm-restart > '$log_dir/controller.stdout.log' 2>&1
"
  sleep 5
  if ! tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
    echo "[$(timestamp)] launch verification failed for $TARGET_SESSION" >&2
    tail -80 "$log_dir/controller.stdout.log" >&2 || true
    exit 1
  fi
  echo "[$(timestamp)] launched $TARGET_SESSION full matrix (30 runs; core seeds 42,13 first) log_dir=$log_dir"
}

wait_for_idle
if ! vllm_ready; then
  echo "[$(timestamp)] vLLM endpoints 8000/8001 not ready — not launching. Start them or drop --skip-vllm-restart." >&2
  exit 1
fi
validate
run_smoke
completed="$(smoke_completed)"
echo "[$(timestamp)] smoke completed=$completed/$SMOKE_EXPECTED"
if [ "$completed" -lt "$SMOKE_EXPECTED" ]; then
  echo "[$(timestamp)] SMOKE FAILED — not launching full matrix. Inspect $SMOKE_LOG_DIR/results.json" >&2
  exit 1
fi
engaged="$(llm_engaged)"
echo "[$(timestamp)] LLM modes placed bids in smoke: $engaged"
if [ "$engaged" != "yes" ]; then
  echo "[$(timestamp)] ENGAGEMENT GATE FAILED — LLM modes placed no bids; not launching full." >&2
  exit 1
fi
launch_full
echo "[$(timestamp)] DONE: smoke passed gate, full matrix launched in tmux session $TARGET_SESSION"
