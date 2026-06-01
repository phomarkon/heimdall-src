#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${HEIMDALL_VLLM_BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BASE_DIR"
export HEIMDALL_VLLM_BASE_DIR="$BASE_DIR"

MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
)

RUN_ID="$(date +%Y%m%d-%H%M%S)"
RESULTS="$BASE_DIR/logs/model_matrix_${RUN_ID}.md"

mkdir -p "$BASE_DIR/logs"

cat > "$RESULTS" <<EOF
# Heimdall vLLM model matrix

Started: $(date)

Scope: smoke/model-ladder compatibility check only. Real AI-society matrices use
HEIMDALL_MAX_MODEL_LEN=16384 and the dual-endpoint layout.

Settings:
- HEIMDALL_MAX_MODEL_LEN=8192
- HEIMDALL_GPU_MEMORY_UTILIZATION=0.45
- HEIMDALL_ATTENTION_BACKEND=TRITON_ATTN
- VLLM_USE_DEEP_GEMM=0
- HEIMDALL_VLLM_EXTRA_ARGS="--enforce-eager --enable-auto-tool-choice --tool-call-parser qwen3_xml"

| Model | Launch | Healthcheck | 5-agent named tool calls | GPU memory after test | Notes |
|---|---:|---:|---:|---|---|
EOF

set_env_value() {
  local key="$1"
  local value="$2"
  python - "$key" "$value" <<'PY'
import sys
import os
from pathlib import Path

key, value = sys.argv[1], sys.argv[2]
path = Path(os.environ.get("HEIMDALL_VLLM_BASE_DIR", "/work/heimdall-vllm")) / ".env"
lines = path.read_text().splitlines()
updated = False
for i, line in enumerate(lines):
    if line.startswith(f"{key}="):
        lines[i] = f"{key}={value}"
        updated = True
        break
if not updated:
    lines.append(f"{key}={value}")
path.write_text("\n".join(lines) + "\n")
PY
}

run_one() {
  local model="$1"
  local safe_model="${model//\//__}"
  local model_log="$BASE_DIR/logs/model_matrix_${RUN_ID}_${safe_model}.log"
  local launch="FAIL"
  local health="FAIL"
  local agents="FAIL"
  local gpu_mem="n/a"
  local notes=""

  echo "=== Testing $model ===" | tee -a "$model_log"

  set_env_value "HEIMDALL_MODEL" "$model"
  # Smoke-only setting for the compatibility ladder. Do not use as the default
  # for real AI-society experiment matrices.
  set_env_value "HEIMDALL_MAX_MODEL_LEN" "8192"
  set_env_value "HEIMDALL_GPU_MEMORY_UTILIZATION" "0.45"
  set_env_value "HEIMDALL_ATTENTION_BACKEND" "TRITON_ATTN"
  set_env_value "VLLM_USE_DEEP_GEMM" "0"
  set_env_value "HEIMDALL_VLLM_EXTRA_ARGS" "\"--enforce-eager --enable-auto-tool-choice --tool-call-parser qwen3_xml\""

  bash scripts/stop_vllm.sh >>"$model_log" 2>&1 || true
  if bash scripts/launch_tmux.sh >>"$model_log" 2>&1; then
    launch="PASS"
  else
    notes="launch command failed"
  fi

  source "$BASE_DIR/.venv/bin/activate"

  if [[ "$launch" == "PASS" ]]; then
    if python scripts/healthcheck_vllm.py >>"$model_log" 2>&1; then
      health="PASS"
    else
      notes="healthcheck failed"
    fi
  fi

  if [[ "$health" == "PASS" ]]; then
    if python tests/test_heimdall_n_agents.py --agents 5 >>"$model_log" 2>&1; then
      agents="PASS"
    else
      notes="5-agent tool-call test failed"
    fi
  fi

  gpu_mem="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r' || true)"
  if [[ -z "$gpu_mem" ]]; then
    gpu_mem="n/a"
  fi

  if [[ -z "$notes" ]]; then
    notes="ok"
  fi

  printf '| `%s` | %s | %s | %s | `%s` | %s |\n' "$model" "$launch" "$health" "$agents" "$gpu_mem" "$notes" >> "$RESULTS"
}

for model in "${MODELS[@]}"; do
  run_one "$model"
done

cat >> "$RESULTS" <<EOF

Finished: $(date)

Per-model logs:

$(ls -1 "$BASE_DIR"/logs/model_matrix_"$RUN_ID"_*.log 2>/dev/null || true)
EOF

cat "$RESULTS"
