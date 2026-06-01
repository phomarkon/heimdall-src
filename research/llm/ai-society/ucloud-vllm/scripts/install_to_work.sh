#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${HEIMDALL_VLLM_INSTALL_DIR:-/work/heimdall-vllm}"
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

log() { printf '\033[36m[ucloud-vllm]\033[0m %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
run() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[dry-run] %q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

log "source: $SOURCE_DIR"
log "target: $TARGET_DIR"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer is intended for a Linux UCloud GPU server." >&2
  exit 1
fi

if ! have nvidia-smi; then
  if [[ "$DRY_RUN" == 1 ]]; then
    log "nvidia-smi not found; continuing because --dry-run was set"
  else
  echo "nvidia-smi not found. Choose a UCloud GPU image/server before installing vLLM." >&2
  exit 1
  fi
else
  log "NVIDIA devices visible:"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
fi

if [[ ! -d /work ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then
    log "/work does not exist; continuing because --dry-run was set"
  else
  echo "/work does not exist. UCloud persistent runtime storage is required." >&2
  exit 1
  fi
else
  df -h /work
fi

if ! have uv; then
  log "uv not found; installing in user space"
  if have curl; then
    run sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    export PATH="$HOME/.local/bin:$PATH"
  else
    run python3 -m pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

log "copying reusable vLLM runtime files"
run mkdir -p "$TARGET_DIR"
for name in configs scripts tests clients; do
  run rm -rf "$TARGET_DIR/$name"
  run cp -R "$SOURCE_DIR/$name" "$TARGET_DIR/$name"
done
run cp "$SOURCE_DIR/SETUP_REPORT.md" "$TARGET_DIR/SETUP_REPORT.md"
run chmod +x "$TARGET_DIR"/scripts/*.sh

if [[ "$DRY_RUN" == 1 ]]; then
  log "dry run complete; no files changed"
  exit 0
fi

cd "$TARGET_DIR"

if [[ ! -f .env ]]; then
  cp configs/vllm.env.example .env
  log "created $TARGET_DIR/.env from example"
else
  log "preserving existing $TARGET_DIR/.env"
fi

log "creating Python 3.12 venv"
uv venv --python 3.12 --seed --managed-python
# shellcheck disable=SC1091
source "$TARGET_DIR/.venv/bin/activate"

log "installing OpenAI client/test dependencies"
uv pip install -U pip setuptools wheel
uv pip install -U openai pydantic httpx python-dotenv rich

log "installing vLLM stable wheel"
if ! uv pip install -U vllm --torch-backend=auto; then
  log "stable vLLM install failed; trying nightly wheel index"
  uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
fi

log "verifying vLLM import and CUDA visibility"
python - <<'PY'
import platform
import torch
import vllm

print("python:", platform.python_version())
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
print("gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("gpu", i, torch.cuda.get_device_name(i))
        print("capability", torch.cuda.get_device_capability(i))
print("vllm:", vllm.__version__)
PY

cat <<EOF

Native Heimdall vLLM runtime installed.

Next:
  cd $TARGET_DIR
  bash scripts/launch_tmux.sh
  tail -f logs/vllm.log
  source .venv/bin/activate
  python scripts/healthcheck_vllm.py
  python tests/test_heimdall_named_tool_call.py
  source scripts/export_env.sh

EOF
