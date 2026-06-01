#!/usr/bin/env bash
# D3 faithfulness matrix, P-wide parallel. Each run still load-balances across all 4 vLLM
# endpoints; running P runs at once fills the B200 batch (latency-bound otherwise). Logs per run.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
LIST="${1:-ai-society/configs/d3-faithfulness-20260524/full.txt}"
P="${2:-4}"
LOGDIR="logs/d3-faithfulness-20260524"
mkdir -p "$LOGDIR"
export PYTHONPATH="ai-society/src:."

run_one() {
  local cfg="$1" rid
  rid="$(basename "$cfg" .yaml)"
  echo "[$(date +%H:%M:%S)] START $rid"
  if timeout 2400 uv run python -m heimdall_ai_society run --config "$cfg" > "$LOGDIR/$rid.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK   $rid"
  else
    echo "[$(date +%H:%M:%S)] FAIL $rid"
  fi
}
export -f run_one
export LOGDIR

mapfile -t cfgs < <(grep -v '^$' "$LIST")
echo "dispatching ${#cfgs[@]} runs, $P-wide"
i=0
for cfg in "${cfgs[@]}"; do
  run_one "$cfg" &
  i=$((i+1))
  if (( i % P == 0 )); then wait -n 2>/dev/null || wait; fi
done
wait
echo "ALL DONE"
