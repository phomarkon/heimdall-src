#!/usr/bin/env bash
# Queue the llm-agency-capture matrix BEHIND the current GPU queue
# (deliberation-s12-and-s06-extended-large -> central-supervisor-s06), then:
#   wait-for-idle -> validate -> 7x 2-tick smoke (restarts vLLM to Qwen3-32B)
#   -> hard gate (all smoke complete + cp13 made autonomous tool calls)
#   -> launch 21-run full matrix + compare table.
# Fail-fast: a broken smoke or a collapsed-agency cp13 smoke stops before the ~4h full run.
set -euo pipefail

cd /home/ucloud/heimdall
export PYTHONPATH=.

MATRIX="llm-agency-capture-20260522"
ROOT="ai-society/configs/$MATRIX"
RUN_ROOT="ai-society/runs/$MATRIX"
GEN="tools/experiments/generate_llm_agency_capture_20260522.py"
TARGET_SESSION="heimdall-agency-capture"
SMOKE_EXPECTED=7
FULL_EXPECTED=21
CHECKS_REQUIRED=2
POLL_SECONDS=300

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launch.lock"
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another agency-capture launcher owns the lock; exiting"
  exit 0
fi

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

queue_busy() {
  pgrep -f "run_long_model_society_matrix.py" >/dev/null && return 0
  tmux has-session -t "=heimdall-delib-s12-s06-large-20260522" 2>/dev/null && return 0
  tmux has-session -t "=heimdall-central-supervisor-s06-chain" 2>/dev/null && return 0
  tmux has-session -t "=heimdall-central-supervisor-s06" 2>/dev/null && return 0
  return 1
}

wait_for_idle() {
  local idle=0
  while true; do
    if queue_busy; then
      idle=0
      echo "[$(timestamp)] queue busy (deliberation/central-supervisor); waiting"
    else
      idle=$((idle + 1))
      echo "[$(timestamp)] queue idle check $idle/$CHECKS_REQUIRED"
      [ "$idle" -ge "$CHECKS_REQUIRED" ] && break
    fi
    sleep "$POLL_SECONDS"
  done
  echo "[$(timestamp)] queue idle confirmed; proceeding"
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
  echo "[$(timestamp)] smoke start (restarts vLLM -> Qwen3-32B) log_dir=$SMOKE_LOG_DIR"
  uv run python ai-society/run_long_model_society_matrix.py \
    --config-list "$ROOT/smoke.txt" \
    --log-dir "$SMOKE_LOG_DIR" \
    --continue-on-failure \
    > "$SMOKE_LOG_DIR/controller.stdout.log" 2>&1 || true
}

smoke_completed() {
  python3 -c "import json; r=json.load(open('$SMOKE_LOG_DIR/results.json')); print(sum(1 for x in r if x.get('ok')))" 2>/dev/null || echo 0
}

cp13_autonomy_ok() {
  python3 - <<'PY'
import glob, json
ok = False
for path in glob.glob('evaluations/smoke-lac-s06-a4-cp13-refine-*/run_summary.json'):
    data = json.load(open(path))
    if (data.get('autonomous_tool_call_rate') or 0) > 0:
        ok = True
print('yes' if ok else 'no')
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
uv run python tools/evaluation/compare_agency_capture.py >> '$log_dir/controller.stdout.log' 2>&1
"
  sleep 5
  if ! tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
    echo "[$(timestamp)] launch verification failed for $TARGET_SESSION" >&2
    tail -80 "$log_dir/controller.stdout.log" >&2 || true
    exit 1
  fi
  echo "[$(timestamp)] launched $TARGET_SESSION full matrix ($FULL_EXPECTED runs) log_dir=$log_dir"
}

wait_for_idle
validate
run_smoke
completed="$(smoke_completed)"
echo "[$(timestamp)] smoke completed=$completed/$SMOKE_EXPECTED"
if [ "$completed" -lt "$SMOKE_EXPECTED" ]; then
  echo "[$(timestamp)] SMOKE FAILED — not launching full matrix. Inspect $SMOKE_LOG_DIR/results.json" >&2
  exit 1
fi
auton="$(cp13_autonomy_ok)"
echo "[$(timestamp)] cp13 smoke autonomous_tool_call_rate>0: $auton"
if [ "$auton" != "yes" ]; then
  echo "[$(timestamp)] AGENCY GATE FAILED — cp13 made no autonomous tool calls; not launching full." >&2
  exit 1
fi
launch_full
echo "[$(timestamp)] DONE: smoke passed gate, full matrix launched in tmux session $TARGET_SESSION"
