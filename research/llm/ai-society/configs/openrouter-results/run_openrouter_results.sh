#!/usr/bin/env bash
set -euo pipefail

ROOT="ai-society/configs/openrouter-results"
LOG_DIR="ai-society/runs/openrouter-results/logs-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"

PYTHONPATH=ai-society/src:. uv run python - <<'PY'
from pathlib import Path
import os
from packages.config import load_project_env

load_project_env(Path("ai-society/configs/openrouter-results/config-list.txt").resolve())
if not os.environ.get("OPENROUTER_API_KEY", "").strip():
    raise SystemExit("OPENROUTER_API_KEY is not set in the process environment or saved .env")
PY

while IFS= read -r config; do
  [[ -z "$config" ]] && continue
  run_id="$(basename "$config" .yaml)"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] running $run_id"
  PYTHONPATH=ai-society/src:. uv run python -m heimdall_ai_society validate-config "$config" > "$LOG_DIR/$run_id.validate.json"
  PYTHONPATH=ai-society/src:. uv run python -m heimdall_ai_society run --config "$config" 2>&1 | tee "$LOG_DIR/$run_id.run.log"
done < "$ROOT/config-list.txt"
