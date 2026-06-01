#!/usr/bin/env bash
# F8b/c/d/e retrain at 20 epochs instead of 5. The headline negative result
# in §8.1 ("more features didn't help at 5 epochs") may flip at proper
# capacity budget. This is the cheap parallelisation while F12-EBM holds
# only ~1 GB of GPU memory.
set -uo pipefail
cd /work/heimdall
mkdir -p logs/multivar20
for MODEL in f8b f8c f8d f8e; do
  for SEED in 13 42 137 1729 31415; do
    echo "[multivar20 $MODEL seed=$SEED] $(date -u +%FT%TZ)"
    uv run python -m heimdall_forecaster.train.run \
      --config apps/forecaster/src/heimdall_forecaster/train/configs/${MODEL}.yaml \
      --seed "$SEED" \
      2>&1 | tee "logs/multivar20/${MODEL}-${SEED}.log" | grep -E "epoch|val_pinball" | tail -5
  done
done
echo "[multivar20] DONE $(date -u +%FT%TZ)"
uv run python tools/finalize_metrics.py 2>&1 | tail -3
PYTHONPATH=. uv run python experiments/build_leaderboard.py 2>&1 | tail -3
