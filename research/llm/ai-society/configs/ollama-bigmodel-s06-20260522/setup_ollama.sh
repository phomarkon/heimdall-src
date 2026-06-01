#!/usr/bin/env bash
set -euo pipefail

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

if ! command -v ollama >/dev/null 2>&1; then
  if [[ "${HEIMDALL_INSTALL_OLLAMA:-0}" == "1" ]]; then
    log "installing Ollama via official installer"
    curl -fsSL https://ollama.com/install.sh | sh
  else
    log "Ollama is not installed. Install it first from https://ollama.com/download, or rerun with HEIMDALL_INSTALL_OLLAMA=1."
    exit 1
  fi
fi

if ! curl -fsS http://127.0.0.1:11434/v1/models >/dev/null 2>&1; then
  log "starting ollama serve in tmux session heimdall-ollama"
  tmux kill-session -t heimdall-ollama >/dev/null 2>&1 || true
  tmux new-session -d -s heimdall-ollama "ollama serve"
  for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:11434/v1/models >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
fi

curl -fsS http://127.0.0.1:11434/v1/models >/dev/null
for model in qwen3:235b qwen2.5:72b-instruct-q3_K_L qwen:110b; do
  log "pulling $model"
  if ollama pull "$model"; then
    log "pulled $model"
  else
    log "pull failed for $model; launcher will skip it"
  fi
done
