#!/usr/bin/env bash
# Kick off the ~9h overnight breadth run (broad-societies-20260525) with telemetry FIRST,
# so GPU energy + vLLM token/latency are captured for the whole batch (not mid-flight).
# Run only after the current batch has freed the 4 GPUs / endpoints.
set -euo pipefail
cd "$(dirname "$0")/../.."
B=ai-society/runs/broad-societies-20260525
mkdir -p "$B"

nohup uv run python tools/observability/gpu_telemetry.py --out "$B/gpu_telemetry.csv" --interval 5 \
  > "$B/telemetry.log" 2>&1 &
echo "gpu_telemetry PID $!"
nohup uv run python tools/observability/vllm_metrics.py --out "$B/vllm_metrics.csv" --interval 5 \
  > "$B/vllm_metrics.log" 2>&1 &
echo "vllm_metrics PID $!"
sleep 6

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=.:ai-society/src \
  nohup uv run python tools/experiments/run_ff_rag_batch.py \
  --list ai-society/configs/broad-societies-20260525/full.txt \
  --out "$B/results.jsonl" > "$B/run.log" 2>&1 &
echo "broad-societies batch PID $!  (72 runs, ~9h; log: $B/run.log)"
