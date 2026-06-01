#!/usr/bin/env bash
# Generic matrix launcher with a HARD P-wide cap via xargs -P (the previous wait -n pool did NOT
# cap concurrency and oversubscribed the GPUs). Each run load-balances across the vLLM endpoints in
# its config. Args: <config-list.txt> <logdir> [P]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
LIST="$1"; LOGDIR="$2"; P="${3:-4}"
mkdir -p "$LOGDIR"
export PYTHONPATH="ai-society/src:."
n=$(grep -c . "$LIST")
echo "dispatching $n runs, HARD $P-wide -> $LOGDIR"
xargs -P "$P" -I{} bash -c '
  c="$1"; rid=$(basename "$c" .yaml)
  echo "[$(date +%H:%M:%S)] START $rid"
  if timeout 2400 uv run python -m heimdall_ai_society run --config "$c" > "'"$LOGDIR"'/$rid.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK   $rid"
  else
    echo "[$(date +%H:%M:%S)] FAIL $rid"
  fi
' _ {} < "$LIST"
echo "ALL DONE"
