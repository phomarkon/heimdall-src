#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${HEIMDALL_VLLM_BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BASE_DIR"

source "$BASE_DIR/.venv/bin/activate"

if [[ -f "$BASE_DIR/.env" ]]; then
  set -a
  source "$BASE_DIR/.env"
  set +a
fi

export HF_HOME="${HF_HOME:-/work/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/work/.cache/vllm}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$BASE_DIR/logs"

HEIMDALL_MODEL="${HEIMDALL_MODEL:-Qwen/Qwen3-0.6B}"
HEIMDALL_VLLM_HOST="${HEIMDALL_VLLM_HOST:-127.0.0.1}"
HEIMDALL_VLLM_PORT="${HEIMDALL_VLLM_PORT:-8000}"
HEIMDALL_VLLM_API_KEY="${HEIMDALL_VLLM_API_KEY:-heimdall-local}"
HEIMDALL_TENSOR_PARALLEL_SIZE="${HEIMDALL_TENSOR_PARALLEL_SIZE:-1}"
HEIMDALL_MAX_MODEL_LEN="${HEIMDALL_MAX_MODEL_LEN:-16384}"
HEIMDALL_GPU_MEMORY_UTILIZATION="${HEIMDALL_GPU_MEMORY_UTILIZATION:-0.80}"
HEIMDALL_DTYPE="${HEIMDALL_DTYPE:-auto}"
HEIMDALL_TRUST_REMOTE_CODE="${HEIMDALL_TRUST_REMOTE_CODE:-0}"
HEIMDALL_ATTENTION_BACKEND="${HEIMDALL_ATTENTION_BACKEND:-TRITON_ATTN}"
HEIMDALL_VLLM_EXTRA_ARGS="${HEIMDALL_VLLM_EXTRA_ARGS:-}"

cmd=(
  vllm serve "$HEIMDALL_MODEL"
  --host "$HEIMDALL_VLLM_HOST"
  --port "$HEIMDALL_VLLM_PORT"
  --api-key "$HEIMDALL_VLLM_API_KEY"
  --dtype "$HEIMDALL_DTYPE"
  --gpu-memory-utilization "$HEIMDALL_GPU_MEMORY_UTILIZATION"
  --max-model-len "$HEIMDALL_MAX_MODEL_LEN"
)

if [[ "$HEIMDALL_ATTENTION_BACKEND" != "auto" && -n "$HEIMDALL_ATTENTION_BACKEND" ]]; then
  cmd+=(--attention-backend "$HEIMDALL_ATTENTION_BACKEND")
fi

if [[ "$HEIMDALL_TENSOR_PARALLEL_SIZE" != "1" ]]; then
  cmd+=(--tensor-parallel-size "$HEIMDALL_TENSOR_PARALLEL_SIZE")
fi

if [[ "$HEIMDALL_TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

echo "Starting vLLM for Heimdall..."
echo "Model: $HEIMDALL_MODEL"
echo "Endpoint: http://$HEIMDALL_VLLM_HOST:$HEIMDALL_VLLM_PORT/v1"
echo "HF_HOME: $HF_HOME"
echo "VLLM_CACHE_ROOT: $VLLM_CACHE_ROOT"
echo "HEIMDALL_ATTENTION_BACKEND: $HEIMDALL_ATTENTION_BACKEND"
echo "VLLM_USE_DEEP_GEMM: $VLLM_USE_DEEP_GEMM"
echo "Base command: ${cmd[*]}"
echo "Extra args: $HEIMDALL_VLLM_EXTRA_ARGS"

# Intentional unquoted extra args so users can pass multiple CLI flags from .env.
exec "${cmd[@]}" $HEIMDALL_VLLM_EXTRA_ARGS
