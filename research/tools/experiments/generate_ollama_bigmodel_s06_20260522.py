from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

MATRIX = "ollama-bigmodel-s06-20260522"
CONFIG_ROOT = Path("ai-society/configs") / MATRIX
RUN_ROOT = Path("ai-society/runs") / MATRIX
CONTEXT_DIR = "data/cache/real_context/april_2026"
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
MODELS = {
    "qwen3-235b": "qwen3:235b",
    "qwen2p5-72b": "qwen2.5:72b-instruct-q3_K_L",
    "qwen110b": "qwen:110b",
}
WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Generate {MATRIX} configs.")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    full = [_payload(model_slug, window_slug, ticks=24, smoke=False) for model_slug in MODELS for window_slug in WINDOWS]
    smoke = [_payload(model_slug, "apr02-0530", ticks=2, smoke=True) for model_slug in MODELS]
    _check(full, smoke)
    if args.check_only:
        print(json.dumps({"full": len(full), "smoke": len(smoke)}, sort_keys=True))
        return 0
    _write(full, smoke)
    print(json.dumps({"config_root": str(CONFIG_ROOT), "full": len(full), "smoke": len(smoke)}, sort_keys=True))
    return 0


def _payload(model_slug: str, window_slug: str, *, ticks: int, smoke: bool) -> dict[str, Any]:
    prefix = "smoke-obm" if smoke else "obm"
    run_id = f"{prefix}-s06-actioncore-{model_slug}-guarded-{window_slug}-seed42-{ticks}"
    return {
        "run_id": run_id,
        "seed": 42,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": 6,
        "ticks": ticks,
        "start_timestamp": WINDOWS[window_slug],
        "forecaster_backend": "f8",
        "forecaster_routing_mode": "persona",
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "market_context": "real",
        "context_dataset_dir": CONTEXT_DIR,
        "cache_refresh": False,
        "tool_mode": "json_response",
        "preprobe_mode": "full",
        "objective": "bid_seeking",
        "final_bid_guard": "simulator_exact_match",
        "safety_toolset": "full",
        "ablation_strategy": "comm_broadcast_digest_priority_calibration",
        "persona_profile": "action_core_8",
        "tool_policy": "asset_simulator_v1",
        "asset_simulator_mode": "scenario_envelope",
        "candidate_sizing_mode": "large",
        "candidate_sizing_max_candidates": 8,
        "candidate_sizing_min_mwh": 0.25,
        "candidate_sizing_cap_fraction": 1.0,
        "output_dir": str(RUN_ROOT),
        "llm": {
            "enabled": True,
            "provider": "ollama",
            "model": MODELS[model_slug],
            "base_url": OLLAMA_BASE_URL,
            "api_key": "ollama",
            "temperature": 0.2,
            "max_tokens": 1000,
            "timeout_seconds": 600,
            "max_concurrency": 2,
            "per_endpoint_max_concurrency": 2,
            "require_served_model_match": True,
            "supports_response_format": True,
            "supports_tools": True,
        },
    }


def _check(full: list[dict[str, Any]], smoke: list[dict[str, Any]]) -> None:
    if len(full) != 9:
        raise RuntimeError(f"expected 9 full configs, got {len(full)}")
    if len(smoke) != 3:
        raise RuntimeError(f"expected 3 smoke configs, got {len(smoke)}")
    run_ids = [payload["run_id"] for payload in [*full, *smoke]]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("duplicate run_id")
    for payload in [*full, *smoke]:
        llm = payload["llm"]
        if llm["provider"] != "ollama" or llm["base_url"] != OLLAMA_BASE_URL:
            raise RuntimeError(f"bad Ollama LLM config: {payload['run_id']}")
        if "base_urls" in llm:
            raise RuntimeError(f"Ollama configs must not set base_urls: {payload['run_id']}")


def _write(full: list[dict[str, Any]], smoke: list[dict[str, Any]]) -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    full_paths = [_write_payload(payload, "full") for payload in full]
    smoke_paths = [_write_payload(payload, "smoke") for payload in smoke]
    (CONFIG_ROOT / "all.txt").write_text("\n".join(str(path) for path in full_paths) + "\n", encoding="utf-8")
    (CONFIG_ROOT / "smoke.txt").write_text("\n".join(str(path) for path in smoke_paths) + "\n", encoding="utf-8")
    for model_slug in MODELS:
        (CONFIG_ROOT / f"{model_slug}-all.txt").write_text(
            "\n".join(str(path) for path in full_paths if f"-{model_slug}-" in path.stem) + "\n",
            encoding="utf-8",
        )
        (CONFIG_ROOT / f"{model_slug}-smoke.txt").write_text(
            "\n".join(str(path) for path in smoke_paths if f"-{model_slug}-" in path.stem) + "\n",
            encoding="utf-8",
        )
    (CONFIG_ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "matrix": MATRIX,
                "created_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
                "full_count": len(full_paths),
                "smoke_count": len(smoke_paths),
                "models": MODELS,
                "windows": WINDOWS,
                "ollama_base_url": OLLAMA_BASE_URL,
                "tool_mode": "json_response",
                "note": "Generated candidates for all target Ollama models; launch script records which models are available after pull/serve checks.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (CONFIG_ROOT / "RUNBOOK.md").write_text(_runbook(), encoding="utf-8")
    setup = CONFIG_ROOT / "setup_ollama.sh"
    setup.write_text(_setup_script(), encoding="utf-8")
    setup.chmod(0o755)
    launcher = CONFIG_ROOT / "launch_ollama_matrix.sh"
    launcher.write_text(_launcher(), encoding="utf-8")
    launcher.chmod(0o755)


def _write_payload(payload: dict[str, Any], split: str) -> Path:
    model_slug = next(slug for slug, model in MODELS.items() if model == payload["llm"]["model"])
    path = CONFIG_ROOT / split / model_slug / f"{payload['run_id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _runbook() -> str:
    return f"""# {MATRIX}

Ollama-backed S06 big-model baseline screen over the three core April windows.

1. Generate/validate configs:
   `uv run python tools/experiments/generate_ollama_bigmodel_s06_20260522.py`
2. Install/start/pull Ollama models:
   `bash {CONFIG_ROOT}/setup_ollama.sh`
3. Launch smoke then full matrix:
   `bash {CONFIG_ROOT}/launch_ollama_matrix.sh`

The launcher writes `available-models.json`, `available-smoke.txt`, and `available-all.txt` after pull/serve checks. It skips models that cannot be pulled or listed by Ollama.
"""


def _setup_script() -> str:
    models = " ".join(MODELS.values())
    return f"""#!/usr/bin/env bash
set -euo pipefail

log() {{ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }}

if ! command -v ollama >/dev/null 2>&1; then
  if [[ "${{HEIMDALL_INSTALL_OLLAMA:-0}}" == "1" ]]; then
    log "installing Ollama via official installer"
    curl -fsSL https://ollama.com/install.sh | sh
  else
    log "Ollama is not installed. Install it first from https://ollama.com/download, or rerun with HEIMDALL_INSTALL_OLLAMA=1."
    exit 1
  fi
fi

if ! curl -fsS {OLLAMA_BASE_URL}/models >/dev/null 2>&1; then
  log "starting ollama serve in tmux session heimdall-ollama"
  tmux kill-session -t heimdall-ollama >/dev/null 2>&1 || true
  tmux new-session -d -s heimdall-ollama "ollama serve"
  for _ in $(seq 1 60); do
    if curl -fsS {OLLAMA_BASE_URL}/models >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
fi

curl -fsS {OLLAMA_BASE_URL}/models >/dev/null
for model in {models}; do
  log "pulling $model"
  if ollama pull "$model"; then
    log "pulled $model"
  else
    log "pull failed for $model; launcher will skip it"
  fi
done
"""


def _launcher() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT={CONFIG_ROOT}
RUN_ROOT={RUN_ROOT}
OLLAMA_BASE_URL={OLLAMA_BASE_URL}
log() {{ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }}

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
models = {json.dumps(MODELS, sort_keys=True)}
available = {{}}
skipped = {{}}
for slug, model in models.items():
    pull = subprocess.run(["ollama", "pull", model], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if pull.returncode != 0:
        skipped[slug] = {{"model": model, "reason": "pull_failed", "log_tail": pull.stdout[-2000:]}}
        continue
    try:
        with urllib.request.urlopen("{OLLAMA_BASE_URL}/models", timeout=10) as response:
            served = {{item["id"] for item in json.loads(response.read().decode("utf-8")).get("data", [])}}
    except Exception as exc:
        skipped[slug] = {{"model": model, "reason": f"models_endpoint_failed: {{exc}}"}}
        continue
    if model in served:
        available[slug] = model
    else:
        skipped[slug] = {{"model": model, "reason": f"not_listed_by_ollama: {{sorted(served)}}"}}

(root / "available-models.json").write_text(json.dumps({{"available": available, "skipped": skipped}}, indent=2, sort_keys=True) + "\\n")
smoke = []
full = []
for slug in available:
    smoke.extend(path for path in (root / f"{{slug}}-smoke.txt").read_text().splitlines() if path)
    full.extend(path for path in (root / f"{{slug}}-all.txt").read_text().splitlines() if path)
(root / "available-smoke.txt").write_text("\\n".join(smoke) + ("\\n" if smoke else ""))
(root / "available-all.txt").write_text("\\n".join(full) + ("\\n" if full else ""))
if not available:
    raise SystemExit("no Ollama target models available; see available-models.json")
print(json.dumps({{"available": available, "smoke": len(smoke), "full": len(full)}}, sort_keys=True))
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
PYTHONPATH=. uv run python ai-society/run_ollama_society_matrix.py \\
  --config-list "$ROOT/available-smoke.txt" \\
  --log-dir "$smoke_log_dir" \\
  --continue-on-failure

python - "$smoke_log_dir/summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
if d.get("completed", 0) < 1 or d.get("failed", 0) != 0:
    raise SystemExit(f"smoke failed: {{d}}")
PY

full_log_dir=$RUN_ROOT/logs-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$full_log_dir"
tmux new-session -d -s heimdall-ollama-bigmodel-s06 \\
  "cd /home/ucloud/heimdall && PYTHONPATH=. uv run python ai-society/run_ollama_society_matrix.py --config-list $ROOT/available-all.txt --log-dir $full_log_dir --continue-on-failure > $full_log_dir.controller.stdout.log 2>&1"
log "launched heimdall-ollama-bigmodel-s06 log_dir=$full_log_dir"
"""


if __name__ == "__main__":
    raise SystemExit(main())
