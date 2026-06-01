#!/usr/bin/env bash
# F7-Optuna best-config 5-seed sweep at full 50-epoch budget (Phase 5 ⇒ Phase 6).
set -uo pipefail
cd /work/heimdall
mkdir -p logs/f7_optuna
for SEED in 13 42 137 1729 31415; do
  echo "[f7-optuna seed=$SEED] $(date -u +%FT%TZ)"
  uv run python -m heimdall_forecaster.train.run \
    --config apps/forecaster/src/heimdall_forecaster/train/configs/f7_optuna.yaml \
    --seed "$SEED" 2>&1 | tee "logs/f7_optuna/seed-$SEED.log" | grep -E "epoch|val_pinball" | tail -3
done
echo "[f7-optuna] DONE $(date -u +%FT%TZ)"
uv run python tools/finalize_metrics.py 2>&1 | tail -3
PYTHONPATH=. uv run python experiments/build_leaderboard.py 2>&1 | tail -2
PYTHONPATH=. uv run python experiments/multi_metric_panel.py 2>&1 | tail -10
PYTHONPATH=. uv run python tools/generate_model_cards.py 2>&1 | tail -2
echo "[f7-optuna] per-seed:"
for s in 13 42 137 1729 31415; do
  uv run python -c "
import json
d=json.load(open('models/forecaster/f7_optuna/seed-$s/metrics.json'))
print(f'f7_optuna seed=$s pinball={d[\"val_pinball_mean\"]:.1f} ACI={d[\"aci_empirical_coverage\"]:.3f}')" 2>/dev/null
done
