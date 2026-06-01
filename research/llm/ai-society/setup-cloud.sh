#!/usr/bin/env bash
# Server-only setup checks for Heimdall AI society on a UCloud Ubuntu B200 box.
# Native vLLM is installed separately into /work by ai-society/ucloud-vllm.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FORCE_CPU_DEV=0
INSTALL_DOCKER=0

for arg in "$@"; do
  case "$arg" in
    --force-cpu-dev) FORCE_CPU_DEV=1 ;;
    --docker) INSTALL_DOCKER=1 ;;
    --start-vllm)
      echo "--start-vllm was removed from the default UCloud path." >&2
      echo "Use: bash ai-society/ucloud-vllm/scripts/install_to_work.sh && cd /work/heimdall-vllm && bash scripts/launch_tmux.sh" >&2
      exit 2
      ;;
    -h|--help)
      sed -n '1,60p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

log() { printf '\033[36m[ai-society]\033[0m %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

if [[ "$(uname -s)" != "Linux" && "$FORCE_CPU_DEV" != 1 ]]; then
  echo "This setup is intended for a Linux UCloud GPU server. Use --force-cpu-dev only for config validation." >&2
  exit 1
fi

if ! have nvidia-smi; then
  if [[ "$FORCE_CPU_DEV" == 1 ]]; then
    log "nvidia-smi not found; continuing because --force-cpu-dev was set"
  else
    echo "nvidia-smi not found. Stop here: choose a GPU image/server or install NVIDIA drivers first." >&2
    exit 1
  fi
else
  log "NVIDIA devices visible:"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
fi

if [[ "$FORCE_CPU_DEV" == 1 ]]; then
  log "CPU-dev mode: validating laptop-safe config only"
  if have uv; then
    PYTHONPATH=. uv run python -m heimdall_ai_society validate-config ai-society/configs/local-dryrun.yaml >/dev/null
  else
    PYTHONPATH=. python -m heimdall_ai_society validate-config ai-society/configs/local-dryrun.yaml >/dev/null
  fi
  log "CPU-dev validation passed"
  exit 0
fi

if [[ ! -d /work ]]; then
  echo "/work does not exist. UCloud persistent runtime storage is required for native vLLM." >&2
  exit 1
fi

log "/work storage:"
df -h /work

if ! have uv; then
  log "uv not found; installing in user space"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
log "uv: $(uv --version)"

if [[ "$INSTALL_DOCKER" == 1 ]]; then
  if have apt-get; then
    log "installing optional Docker/NVIDIA runtime"
    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg lsb-release

    if ! have docker; then
      sudo install -m 0755 -d /etc/apt/keyrings
      curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
      sudo chmod a+r /etc/apt/keyrings/docker.gpg
      . /etc/os-release
      echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
      sudo apt-get update -y
      sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    fi

    if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
      sudo apt-get update -y
      sudo apt-get install -y nvidia-container-toolkit
    fi
  fi

  if have nvidia-ctk && have docker; then
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker || true
  fi
fi

mkdir -p ai-society/runs
MANIFEST="ai-society/runs/cloud-runtime-manifest.json"
{
  echo "{"
  echo "  \"created_at_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"git_commit\": \"$(git rev-parse --short HEAD 2>/dev/null || echo unknown)\","
  echo "  \"native_vllm_dir\": \"/work/heimdall-vllm\","
  echo "  \"default_model_id\": \"Qwen/Qwen3-32B\","
  echo "  \"gpu_summary\": \"$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | tr '\n' ';' | sed 's/"/\\"/g')\""
  echo "}"
} > "$MANIFEST"
log "wrote $MANIFEST"

cat <<'EOF'

AI society cloud checks complete.

Next native UCloud vLLM commands:
  bash ai-society/ucloud-vllm/scripts/install_to_work.sh
  cd /work/heimdall-vllm
  bash scripts/launch_tmux.sh
  tail -f logs/vllm.log
  source .venv/bin/activate
  python scripts/healthcheck_vllm.py
  python tests/test_heimdall_named_tool_call.py
  source scripts/export_env.sh

Optional Docker path, only on UCloud VMs where Docker works:
  bash ai-society/setup-cloud.sh --docker
  docker compose -f ai-society/docker-compose.yml up -d

EOF
