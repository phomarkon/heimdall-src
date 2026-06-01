#!/usr/bin/env bash
# Test the undertraining hypothesis: F8b at 100 epochs vs 20-epoch baseline.
# Train all 5 frozen seeds sequentially. Save under f8b_100ep/ dir to avoid
# overwriting the 20-epoch baseline.
set -uo pipefail
cd /work/heimdall
mkdir -p logs/f8b_100ep

# Patch the trainer config name so output goes to a fresh dir.
# We do this by overriding via run.py --seed and renaming in-place won't work.
# Simpler: backup existing seed dirs, run, then move outputs.

# Save current 20-epoch results
mkdir -p models/forecaster/f8b_20ep
for s in 13 42 137 1729 31415; do
  if [ -d "models/forecaster/f8b/seed-$s" ]; then
    cp -r "models/forecaster/f8b/seed-$s" "models/forecaster/f8b_20ep/seed-$s"
  fi
done

# Train F8b at 100 epochs (config edited in place)
for s in 13 42 137 1729 31415; do
  echo "[f8b-100ep seed=$s] $(date -u +%FT%TZ)"
  uv run python -m heimdall_forecaster.train.run \
    --config apps/forecaster/src/heimdall_forecaster/train/configs/f8b.yaml \
    --seed "$s" 2>&1 | tee "logs/f8b_100ep/seed-$s.log" | grep -E "epoch|val_pinball" | tail -3
done
echo "[f8b-100ep] DONE $(date -u +%FT%TZ)"
uv run python tools/finalize_metrics.py 2>&1 | tail -3
PYTHONPATH=. uv run python experiments/build_leaderboard.py 2>&1 | tail -2
