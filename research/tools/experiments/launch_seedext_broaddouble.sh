#!/usr/bin/env bash
# Seed-extension run (2026-05-26): rerun the broad-double slab on seeds 13 and 137
# to add error bars to the seed-42 numbers from overnight-matrix-20260525.
# 36 configs (6 windows × 3 arms × 2 seeds). Single process, balances across the
# 4 vLLM endpoints already pinned in each config. Expected ~6-8h.
set -euo pipefail
cd "$(dirname "$0")/../.."
B=ai-society/runs/seedext-broaddouble-20260526
mkdir -p "$B"

nohup uv run python tools/observability/gpu_telemetry.py --out "$B/gpu_telemetry.csv" --interval 5 \
  > "$B/telemetry.log" 2>&1 &
echo "gpu_telemetry PID $!"
nohup uv run python tools/observability/vllm_metrics.py --out "$B/vllm_metrics.csv" --interval 5 \
  > "$B/vllm_metrics.log" 2>&1 &
echo "vllm_metrics PID $!"
sleep 4

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=.:ai-society/src \
  nohup uv run python tools/experiments/run_ff_rag_batch.py \
  --list ai-society/configs/seedext-broaddouble-20260526/full.txt \
  --out "$B/results.jsonl" > "$B/run.log" 2>&1 &
echo "seedext batch PID $!  (36 runs, ~6-8h; log: $B/run.log)"
