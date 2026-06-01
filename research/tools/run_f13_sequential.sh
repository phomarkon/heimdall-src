#!/usr/bin/env bash
# F13 5 seeds sequentially. Parallel run failed due to MLflow file-store
# concurrent-write contention. Sequential is slower but reliable.
set -uo pipefail
cd /work/heimdall
mkdir -p logs/f13_seq
for SEED in 13 42 137 1729 31415; do
  echo "[f13-seq seed=$SEED] $(date -u +%FT%TZ)"
  uv run python -m heimdall_forecaster.train.run \
    --config apps/forecaster/src/heimdall_forecaster/train/configs/f13.yaml \
    --seed "$SEED" 2>&1 | tee "logs/f13_seq/seed-$SEED.log" | grep -E "epoch|val_pinball|FATAL" | tail -3
done
echo "[f13-seq] DONE $(date -u +%FT%TZ)"
uv run python tools/finalize_metrics.py 2>&1 | tail -3
PYTHONPATH=. uv run python experiments/build_leaderboard.py 2>&1 | tail -2

# Print F13 final
for s in 13 42 137 1729 31415; do
  uv run python -c "
import json
try:
  d = json.load(open('models/forecaster/f13/seed-$s/metrics.json'))
  print(f'F13 seed=$s pinball={d[\"val_pinball_mean\"]:.1f} ACI={d[\"aci_empirical_coverage\"]:.3f}')
except Exception as e: print(f'F13 seed=$s err {e}')
" 2>/dev/null
done
