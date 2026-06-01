#!/usr/bin/env bash
# Run the llm-value-fairtest matrix NOW (queue already idle; no chaining/wait):
#   validate -> 5x 2-tick smoke -> gate (all complete + LLM placed bids) -> 30-run full.
# Reuses the persistent Qwen3-32B vLLM on 8000/8001 (--skip-vllm-restart).
set -euo pipefail

cd /home/ucloud/heimdall
export PYTHONPATH=.

MATRIX="llm-value-fairtest-20260522"
ROOT="ai-society/configs/$MATRIX"
RUN_ROOT="ai-society/runs/$MATRIX"
GEN="tools/experiments/generate_llm_value_fairtest_20260522.py"
SMOKE_EXPECTED=5
API_KEY="heimdall-local"

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launch.lock"
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another fairtest launcher owns the lock; exiting"; exit 0
fi
timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

vllm_ready() {
  curl -fsS -m 5 -H "Authorization: Bearer $API_KEY" http://127.0.0.1:8000/v1/models >/dev/null 2>&1 || return 1
  curl -fsS -m 5 -H "Authorization: Bearer $API_KEY" http://127.0.0.1:8001/v1/models >/dev/null 2>&1 || return 1
  return 0
}

if ! vllm_ready; then
  echo "[$(timestamp)] vLLM 8000/8001 not ready — aborting." >&2; exit 1
fi

uv run python "$GEN" --check-only >/dev/null
while read -r cfg; do [ -z "$cfg" ] || uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null; done < "$ROOT/smoke.txt"
while read -r cfg; do [ -z "$cfg" ] || uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null; done < "$ROOT/all.txt"
echo "[$(timestamp)] configs validated"

SMOKE_LOG_DIR="$RUN_ROOT/logs-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$SMOKE_LOG_DIR"
printf '%s\n' "$SMOKE_LOG_DIR" > "$ROOT/latest-smoke-log-dir.txt"
echo "[$(timestamp)] smoke start (reusing persistent vLLM) log_dir=$SMOKE_LOG_DIR"
uv run python ai-society/run_long_model_society_matrix.py \
  --config-list "$ROOT/smoke.txt" --log-dir "$SMOKE_LOG_DIR" \
  --continue-on-failure --skip-vllm-restart \
  > "$SMOKE_LOG_DIR/controller.stdout.log" 2>&1 || true

completed="$(python3 -c "import json; r=json.load(open('$SMOKE_LOG_DIR/results.json')); print(sum(1 for x in r if x.get('ok')))" 2>/dev/null || echo 0)"
echo "[$(timestamp)] smoke completed=$completed/$SMOKE_EXPECTED"
if [ "$completed" -lt "$SMOKE_EXPECTED" ]; then
  echo "[$(timestamp)] SMOKE FAILED — not launching full. Inspect $SMOKE_LOG_DIR/results.json" >&2; exit 1
fi
engaged="$(python3 - <<'PY'
import glob, json
total = 0
for mode in ("selector", "comm", "memory", "cp12"):
    for path in glob.glob(f"evaluations/smoke-lvf-s06-{mode}-*/run_summary.json"):
        try: total += int(json.load(open(path)).get("bid_action_count") or 0)
        except Exception: pass
print("yes" if total > 0 else "no")
PY
)"
echo "[$(timestamp)] LLM modes placed bids in smoke: $engaged"
if [ "$engaged" != "yes" ]; then
  echo "[$(timestamp)] ENGAGEMENT GATE FAILED — LLM placed no bids; not launching full." >&2; exit 1
fi

LOG_DIR="$RUN_ROOT/logs-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"
printf '%s\n' "$LOG_DIR" > "$ROOT/latest-log-dir.txt"
echo "[$(timestamp)] FULL matrix start (30 runs; core seeds 42,13 first) log_dir=$LOG_DIR"
uv run python ai-society/run_long_model_society_matrix.py \
  --config-list "$ROOT/all.txt" --log-dir "$LOG_DIR" \
  --continue-on-failure --skip-vllm-restart \
  > "$LOG_DIR/controller.stdout.log" 2>&1
echo "[$(timestamp)] DONE: full matrix complete. log_dir=$LOG_DIR"
