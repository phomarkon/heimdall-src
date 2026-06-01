#!/usr/bin/env bash
# Train F11 across all 5 frozen seeds sequentially (shared B200 GPU).
set -euo pipefail
cd /work/heimdall
mkdir -p logs/f11
for SEED in 13 42 137 1729 31415; do
  echo "[f11] starting seed=$SEED at $(date -u +%FT%TZ)"
  uv run python -m heimdall_forecaster.train.run \
      --config apps/forecaster/src/heimdall_forecaster/train/configs/f11.yaml \
      --seed "$SEED" 2>&1 | tee "logs/f11/seed-$SEED.log"
  echo "[f11] done seed=$SEED at $(date -u +%FT%TZ)"
done
echo "[f11] ALL DONE at $(date -u +%FT%TZ)"
