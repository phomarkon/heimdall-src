#!/usr/bin/env bash
set -euo pipefail

ROOT=ai-society/configs/ollama-bigmodel-s06-20260522
RUN_ROOT=ai-society/runs/ollama-bigmodel-s06-20260522
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

uv run python tools/experiments/generate_ollama_bigmodel_s06_20260522.py --check-only
if ! command -v ollama >/dev/null 2>&1; then
  log "Ollama is not installed; run $ROOT/setup_ollama.sh after installing Ollama."
  exit 1
fi
if ! curl -fsS "$OLLAMA_BASE_URL/models" >/dev/null 2>&1; then
  log "Ollama is not serving; run $ROOT/setup_ollama.sh"
  exit 1
fi

python - "$ROOT" <<'PY'
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

root = Path(sys.argv[1])
models = {"qwen110b": "qwen:110b", "qwen2p5-72b": "qwen2.5:72b-instruct-q3_K_L", "qwen3-235b": "qwen3:235b"}
available = {}
skipped = {}
for slug, model in models.items():
    pull = subprocess.run(["ollama", "pull", model], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if pull.returncode != 0:
        skipped[slug] = {"model": model, "reason": "pull_failed", "log_tail": pull.stdout[-2000:]}
        continue
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/v1/models", timeout=10) as response:
            served = {item["id"] for item in json.loads(response.read().decode("utf-8")).get("data", [])}
    except Exception as exc:
        skipped[slug] = {"model": model, "reason": f"models_endpoint_failed: {exc}"}
        continue
    if model in served:
        available[slug] = model
    else:
        skipped[slug] = {"model": model, "reason": f"not_listed_by_ollama: {sorted(served)}"}

(root / "available-models.json").write_text(json.dumps({"available": available, "skipped": skipped}, indent=2, sort_keys=True) + "\n")
smoke = []
full = []
for slug in available:
    smoke.extend(path for path in (root / f"{slug}-smoke.txt").read_text().splitlines() if path)
    full.extend(path for path in (root / f"{slug}-all.txt").read_text().splitlines() if path)
(root / "available-smoke.txt").write_text("\n".join(smoke) + ("\n" if smoke else ""))
(root / "available-all.txt").write_text("\n".join(full) + ("\n" if full else ""))
if not available:
    raise SystemExit("no Ollama target models available; see available-models.json")
print(json.dumps({"available": available, "smoke": len(smoke), "full": len(full)}, sort_keys=True))
PY

while read -r cfg; do
  [[ -z "$cfg" ]] && continue
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
done < "$ROOT/available-smoke.txt"
while read -r cfg; do
  [[ -z "$cfg" ]] && continue
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
done < "$ROOT/available-all.txt"

smoke_log_dir=$RUN_ROOT/logs-smoke-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$smoke_log_dir"
PYTHONPATH=. uv run python ai-society/run_ollama_society_matrix.py \
  --config-list "$ROOT/available-smoke.txt" \
  --log-dir "$smoke_log_dir" \
  --continue-on-failure

python - "$smoke_log_dir/summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
if d.get("completed", 0) < 1 or d.get("failed", 0) != 0:
    raise SystemExit(f"smoke failed: {d}")
PY

full_log_dir=$RUN_ROOT/logs-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$full_log_dir"
tmux new-session -d -s heimdall-ollama-bigmodel-s06 \
  "cd /home/ucloud/heimdall && PYTHONPATH=. uv run python ai-society/run_ollama_society_matrix.py --config-list $ROOT/available-all.txt --log-dir $full_log_dir --continue-on-failure > $full_log_dir.controller.stdout.log 2>&1"
log "launched heimdall-ollama-bigmodel-s06 log_dir=$full_log_dir"
