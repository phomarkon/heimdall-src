#!/usr/bin/env bash
# D3 faithfulness matrix launcher. Runs every config in full.txt sequentially; each run
# load-balances across all 4 vLLM endpoints (8000-8003). Logs per run; continues on failure.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

LIST="${1:-ai-society/configs/d3-faithfulness-20260524/full.txt}"
LOGDIR="logs/d3-faithfulness-20260524"
mkdir -p "$LOGDIR"
export PYTHONPATH="ai-society/src:."

n=0; ok=0; fail=0
while IFS= read -r cfg; do
  [ -z "$cfg" ] && continue
  n=$((n+1))
  rid="$(basename "$cfg" .yaml)"
  echo "[$(date +%H:%M:%S)] ($n) running $rid"
  if timeout 1800 uv run python -m heimdall_ai_society run --config "$cfg" \
        > "$LOGDIR/$rid.log" 2>&1; then
    ok=$((ok+1)); echo "    OK $rid"
  else
    fail=$((fail+1)); echo "    FAIL $rid (see $LOGDIR/$rid.log)"
  fi
done < "$LIST"
echo "DONE: $ok ok / $fail fail / $n total"
