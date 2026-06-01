from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

MATRIX = "qwen-large-baseline-s06-20260522"
CONFIG_ROOT = Path("ai-society/configs") / MATRIX
RUN_ROOT = Path("ai-society/runs") / MATRIX
CONTEXT_DIR = "data/cache/real_context/april_2026"
UPSTREAM_RUN_ROOT = Path("ai-society/runs/high-fill-llm-s06-20260522")
UPSTREAM_EXPECTED_ROWS = 12

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}

MODELS = {
    "q120": "Qwen/Qwen3-120B-A3B",
    "q72": "Qwen/Qwen2.5-72B-Instruct",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Generate {MATRIX} configs.")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    full, smoke = _payloads()
    _check(full, smoke)
    if args.check_only:
        print(json.dumps({"full": len(full), "smoke": len(smoke)}, sort_keys=True))
        return 0
    _write(full, smoke)
    return 0


def _payloads() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full = []
    smoke = []
    for model_slug in ["q120", "q72"]:
        for window_slug, start in WINDOWS.items():
            full.append(_payload(model_slug, window_slug, start, ticks=24, smoke=False))
        smoke.append(_payload(model_slug, "apr02-0530", WINDOWS["apr02-0530"], ticks=2, smoke=True))
    return full, smoke


def _payload(model_slug: str, window_slug: str, start: str, *, ticks: int, smoke: bool) -> dict[str, Any]:
    prefix = "smoke-qlb" if smoke else "qlb"
    run_id = f"{prefix}-s06-actioncore-baseline-{window_slug}-seed42-{ticks}-{model_slug}"
    return {
        "run_id": run_id,
        "seed": 42,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": 6,
        "ticks": ticks,
        "start_timestamp": start,
        "forecaster_backend": "f8",
        "forecaster_routing_mode": "persona",
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "market_context": "real",
        "context_dataset_dir": CONTEXT_DIR,
        "cache_refresh": False,
        "tool_mode": "openai_tools",
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
            "model": MODELS[model_slug],
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "api_key": "heimdall-local",
            "temperature": 0.2,
            "max_tokens": 1000,
            "timeout_seconds": 300,
            "max_concurrency": 4,
            "per_endpoint_max_concurrency": 2,
        },
    }


def _check(full: list[dict[str, Any]], smoke: list[dict[str, Any]]) -> None:
    if len(full) != 6:
        raise RuntimeError(f"expected 6 full configs, got {len(full)}")
    if len(smoke) != 2:
        raise RuntimeError(f"expected 2 smoke configs, got {len(smoke)}")
    run_ids = [payload["run_id"] for payload in [*full, *smoke]]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("duplicate run_id")
    if {payload["candidate_sizing_mode"] for payload in full} != {"large"}:
        raise RuntimeError("large candidate sizing not set")


def _write(full: list[dict[str, Any]], smoke: list[dict[str, Any]]) -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    full_paths = [_write_payload(payload, "full") for payload in full]
    smoke_paths = [_write_payload(payload, "smoke") for payload in smoke]
    (CONFIG_ROOT / "all.txt").write_text("\n".join(str(path) for path in full_paths) + "\n", encoding="utf-8")
    (CONFIG_ROOT / "smoke.txt").write_text("\n".join(str(path) for path in smoke_paths) + "\n", encoding="utf-8")
    (CONFIG_ROOT / "q72-all.txt").write_text(
        "\n".join(str(path) for path in full_paths if "-q72" in path.stem) + "\n",
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
                "question": "Does a larger Qwen model improve baseline scenario-envelope S06 results with large candidate sizing?",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (CONFIG_ROOT / "RUNBOOK.md").write_text(_runbook(), encoding="utf-8")
    launcher = CONFIG_ROOT / "launch_after_high_fill.sh"
    launcher.write_text(_launcher(), encoding="utf-8")
    launcher.chmod(0o755)


def _write_payload(payload: dict[str, Any], split: str) -> Path:
    model_slug = "q120" if payload["llm"]["model"] == MODELS["q120"] else "q72"
    path = CONFIG_ROOT / split / model_slug / f"{payload['run_id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _runbook() -> str:
    return f"""# {MATRIX}

S06 baseline model-scale matrix over the three core windows.

- q120 attempts `{MODELS["q120"]}` first.
- q72 runs `{MODELS["q72"]}` as fallback/comparison.
- Full launcher waits for high-fill to complete 12/12 cleanly.
"""


def _launcher() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ucloud/heimdall
export PYTHONPATH=.

log() {{ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }}

while true; do
  summary=$(ls -td {UPSTREAM_RUN_ROOT}/logs-*/summary.json 2>/dev/null | head -n 1 || true)
  if [[ -n "$summary" ]]; then
    read -r completed failed < <(python - "$summary" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print(d.get("completed", 0), d.get("failed", 0))
PY
)
    if [[ "$completed" == "{UPSTREAM_EXPECTED_ROWS}" && "$failed" == "0" ]]; then
      log "upstream high-fill complete with $completed successful rows"
      break
    fi
    log "waiting for high-fill matrix (running:$completed failed:$failed)"
  else
    log "waiting for high-fill matrix (missing-results)"
  fi
  sleep 300
done

uv run python tools/experiments/generate_qwen_large_baseline_s06_20260522.py --check-only
while read -r cfg; do
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
done < {CONFIG_ROOT}/smoke.txt
while read -r cfg; do
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
done < {CONFIG_ROOT}/all.txt

log_dir={RUN_ROOT}/logs-$(date -u +%Y%m%dT%H%M%SZ)
tmux new-session -d -s heimdall-qwen-large-baseline-s06 \\
  "cd /home/ucloud/heimdall && PYTHONPATH=. uv run python ai-society/run_long_model_society_matrix.py --config-list {CONFIG_ROOT}/all.txt --log-dir $log_dir --continue-on-failure --health-timeout-seconds 1200 > $log_dir.controller.stdout.log 2>&1"
log "launched heimdall-qwen-large-baseline-s06 log_dir=$log_dir"
"""


if __name__ == "__main__":
    raise SystemExit(main())
