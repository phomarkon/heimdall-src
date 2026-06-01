#!/usr/bin/env bash
set -euo pipefail
cd /home/ucloud/heimdall
export PYTHONPATH=.

uv run python tools/experiments/generate_high_fill_llm_s06_20260522.py --check-only
while read -r cfg; do
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
done < ai-society/configs/high-fill-llm-s06-20260522/smoke.txt
while read -r cfg; do
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
done < ai-society/configs/high-fill-llm-s06-20260522/all.txt

smoke_log_dir=ai-society/runs/high-fill-llm-s06-20260522/logs-smoke-$(date -u +%Y%m%dT%H%M%SZ)
uv run python ai-society/run_long_model_society_matrix.py \
  --config-list ai-society/configs/high-fill-llm-s06-20260522/smoke.txt \
  --log-dir "$smoke_log_dir" \
  --skip-vllm-restart

full_log_dir=ai-society/runs/high-fill-llm-s06-20260522/logs-$(date -u +%Y%m%dT%H%M%SZ)
tmux new-session -d -s heimdall-high-fill-llm-s06 \
  "cd /home/ucloud/heimdall && PYTHONPATH=. uv run python ai-society/run_long_model_society_matrix.py --config-list ai-society/configs/high-fill-llm-s06-20260522/all.txt --log-dir $full_log_dir --skip-vllm-restart"
echo "launched heimdall-high-fill-llm-s06 log_dir=$full_log_dir"
