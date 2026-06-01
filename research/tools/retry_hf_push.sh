#!/usr/bin/env bash
# Retry HF push every 10 min until create_commit succeeds or 12 attempts elapse.
# Each call to push_hf.py uses one batched commit, so quota refills steadily.
set -uo pipefail
cd /work/heimdall
: "${HF_TOKEN:?set HF_TOKEN in the environment before running}"
mkdir -p logs
for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
  echo "[retry-hf] attempt=$attempt at $(date -u +%FT%TZ)"
  out=$(uv run python tools/push_hf.py 2>&1 | tail -3)
  echo "$out"
  if echo "$out" | grep -q "uploaded ->"; then
    echo "[retry-hf] SUCCESS at $(date -u +%FT%TZ) attempt=$attempt"
    exit 0
  fi
  if echo "$out" | grep -q "FALLBACK"; then
    echo "[retry-hf] still rate-limited; sleeping 600s ..."
    sleep 600
  else
    echo "[retry-hf] unknown state; sleeping 300s ..."
    sleep 300
  fi
done
echo "[retry-hf] EXHAUSTED 12 attempts; HF push still blocked"
exit 1
