#!/usr/bin/env bash
set -euo pipefail
if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t heimdall-vllm 2>/dev/null || true
fi
pkill -f "vllm serve" || true
echo "Stopped vLLM if it was running."
