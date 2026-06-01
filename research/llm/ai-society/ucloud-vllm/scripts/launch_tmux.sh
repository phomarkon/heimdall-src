#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="${HEIMDALL_VLLM_BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BASE_DIR"
mkdir -p "$BASE_DIR/logs"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Run this instead:"
  echo "cd $BASE_DIR && bash scripts/start_vllm.sh"
  exit 1
fi

tmux has-session -t heimdall-vllm 2>/dev/null && {
  echo "tmux session heimdall-vllm already exists."
  echo "Attach with: tmux attach -t heimdall-vllm"
  exit 0
}

tmux new-session -d -s heimdall-vllm "cd $BASE_DIR && bash scripts/start_vllm.sh > logs/vllm.log 2>&1"
echo "Started tmux session heimdall-vllm"
echo "Logs: tail -f $BASE_DIR/logs/vllm.log"
echo "Attach: tmux attach -t heimdall-vllm"
