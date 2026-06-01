#!/usr/bin/env bash
# Run after F0/F7/F8 (seed_sweep) and F11 have completed all 5 seeds.
# Builds: F3 ensemble, F3-Lite (LSTM), F4 MC-Dropout, F9 leaderboard eval,
# F10 full val, leaderboard, then stages git + (separately) HF push.
set -euo pipefail
cd /work/heimdall
mkdir -p logs

echo "[post-train] $(date -u +%FT%TZ) starting"

# --- F3-Lite (LSTM DeepAR) all 5 seeds ---
echo "[post-train] F3-Lite (LSTM DeepAR) ..."
uv run python experiments/seed_sweep.py --models f3 2>&1 | tee logs/seed_sweep_f3.log | tail -20

# --- F3 ensemble (aggregation over F7 seeds) ---
echo "[post-train] F3 deep ensemble ..."
uv run python -m heimdall_forecaster.train.f3_ensemble 2>&1 | tee logs/f3_ensemble.log | tail -20

# --- F4 MC-Dropout K=30 over F7 backbones, all 5 seeds ---
echo "[post-train] F4 MC-Dropout K=30 ..."
uv run python -m heimdall_forecaster.train.f4_mc_dropout 2>&1 | tee logs/f4_mc_dropout.log | tail -20

# --- F9 TimesFM full-val 5-seed (deterministic; one run replicated) ---
echo "[post-train] F9 TimesFM full-val leaderboard eval ..."
PYTHONPATH=. uv run python experiments/eval_f9_timesfm_zoo.py --backend gpu 2>&1 | tee logs/f9_zoo.log | tail -30

# --- F10 Chronos-Bolt full val, all 5 seeds (deterministic) ---
echo "[post-train] F10 Chronos-Bolt full val ..."
PYTHONPATH=. uv run python experiments/eval_f10_chronos_bolt.py \
    --model amazon/chronos-bolt-base --seeds 13 42 137 1729 31415 \
    2>&1 | tee logs/f10_full.log | tail -30

# --- Finalize metrics.json for any model that lacks it (esp. F11) ---
echo "[post-train] finalizing metrics.json for run.py-trained models ..."
uv run python tools/finalize_metrics.py 2>&1 | tee logs/finalize_metrics.log | tail -20

# --- Leaderboard rebuild ---
echo "[post-train] rebuilding leaderboard ..."
PYTHONPATH=. uv run python experiments/build_leaderboard.py 2>&1 | tee logs/leaderboard.log | tail -10

echo "[post-train] DONE at $(date -u +%FT%TZ)"
