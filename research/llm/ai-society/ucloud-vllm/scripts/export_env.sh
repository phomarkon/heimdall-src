#!/usr/bin/env bash
BASE_DIR="${HEIMDALL_VLLM_BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "$BASE_DIR/.env" ]]; then
  set -a
  source "$BASE_DIR/.env"
  set +a
fi
export OPENAI_BASE_URL="http://${HEIMDALL_VLLM_HOST:-127.0.0.1}:${HEIMDALL_VLLM_PORT:-8000}/v1"
export OPENAI_API_KEY="${HEIMDALL_VLLM_API_KEY:-heimdall-local}"
export HEIMDALL_LLM_MODEL="${HEIMDALL_MODEL:-Qwen/Qwen3-0.6B}"
echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "OPENAI_API_KEY=$OPENAI_API_KEY"
echo "HEIMDALL_LLM_MODEL=$HEIMDALL_LLM_MODEL"
