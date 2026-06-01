#!/usr/bin/env bash
# Run F13 5 seeds in PARALLEL on B200. Each process uses ~2 GB GPU mem;
# 5 in parallel = ~10 GB, plus F8b 100-epoch concurrent. Plenty of headroom
# on B200's 183 GB.
set -uo pipefail
cd /work/heimdall
mkdir -p logs/f13
PIDS=()
for SEED in 13 42 137 1729 31415; do
  echo "[f13 seed=$SEED] launching $(date -u +%FT%TZ)"
  (uv run python -m heimdall_forecaster.train.run \
    --config apps/forecaster/src/heimdall_forecaster/train/configs/f13.yaml \
    --seed "$SEED" > "logs/f13/seed-$SEED.log" 2>&1) &
  PIDS+=($!)
done
echo "[f13] all 5 launched in parallel: PIDs=${PIDS[*]}"
for pid in "${PIDS[@]}"; do
  wait "$pid"
  echo "[f13] PID $pid finished $(date -u +%FT%TZ)"
done
echo "[f13] ALL 5 SEEDS DONE $(date -u +%FT%TZ)"
uv run python tools/finalize_metrics.py 2>&1 | tail -3
PYTHONPATH=. uv run python experiments/build_leaderboard.py 2>&1 | tail -2
