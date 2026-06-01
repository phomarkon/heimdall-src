#!/usr/bin/env bash
# Train F8b (rich, 21 feats) + F8c (kitchen-sink, 39 feats) + F8d (XAI lean)
# + F8e across the 5 frozen seeds, then rebuild leaderboard.
set -uo pipefail
cd /work/heimdall
mkdir -p logs/multivar
for MODEL in f8b f8c f8d f8e; do
  for SEED in 13 42 137 1729 31415; do
    echo "[multivar] $MODEL seed=$SEED at $(date -u +%FT%TZ)"
    uv run python -m heimdall_forecaster.train.run \
      --config apps/forecaster/src/heimdall_forecaster/train/configs/${MODEL}.yaml \
      --seed "$SEED" 2>&1 | tee "logs/multivar/${MODEL}-${SEED}.log" | tail -3
  done
done
echo "[multivar] DONE $(date -u +%FT%TZ)"
uv run python tools/finalize_metrics.py 2>&1 | tail -5
PYTHONPATH=. uv run python experiments/build_leaderboard.py 2>&1 | tail -3
