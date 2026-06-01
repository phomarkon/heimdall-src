from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

MATRIX = "high-fill-llm-s06-20260522"
CONFIG_ROOT = Path("ai-society/configs") / MATRIX
RUN_ROOT = Path("ai-society/runs") / MATRIX
CONTEXT_DIR = "data/cache/real_context/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}

VARIANTS = {
    "deterministic-highfill": {
        "chooser_mode": "deterministic_high_fill_accepted",
        "ablation_strategy": "comm_broadcast_digest_priority_calibration",
        "llm_enabled": False,
        "memory_enabled": False,
    },
    "llm-fill-selector": {
        "chooser_mode": "llm_fill_selector",
        "ablation_strategy": "comm_broadcast_digest_priority_calibration",
        "llm_enabled": True,
        "memory_enabled": False,
    },
    "llm-fill-selector-retry": {
        "chooser_mode": "llm_fill_selector",
        "ablation_strategy": "comm_retry_council",
        "llm_enabled": True,
        "memory_enabled": False,
    },
    "llm-fill-selector-memory-v2": {
        "chooser_mode": "llm_fill_selector",
        "ablation_strategy": "comm_retry_council",
        "llm_enabled": True,
        "memory_enabled": True,
    },
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
    for window_slug, start in WINDOWS.items():
        for variant_slug, variant in VARIANTS.items():
            full.append(_payload(window_slug, start, variant_slug, variant, ticks=24, smoke=False))
            if window_slug == "apr02-0530":
                smoke.append(_payload(window_slug, start, variant_slug, variant, ticks=2, smoke=True))
    return full, smoke


def _payload(window_slug: str, start: str, variant_slug: str, variant: dict[str, Any], *, ticks: int, smoke: bool) -> dict[str, Any]:
    prefix = "smoke-hfl" if smoke else "hfl"
    run_id = f"{prefix}-s06-actioncore-{variant_slug}-{window_slug}-seed42-{ticks}-q32"
    memory_enabled = bool(variant["memory_enabled"])
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
        "chooser_mode": variant["chooser_mode"],
        "verifier_mode": "simulator",
        "market_context": "real",
        "context_dataset_dir": CONTEXT_DIR,
        "cache_refresh": False,
        "tool_mode": "openai_tools",
        "preprobe_mode": "full",
        "objective": "bid_seeking",
        "final_bid_guard": "simulator_exact_match",
        "safety_toolset": "full",
        "ablation_strategy": variant["ablation_strategy"],
        "persona_profile": "action_core_8",
        "tool_policy": "asset_simulator_v1",
        "asset_simulator_mode": "scenario_envelope",
        "candidate_sizing_mode": "medium",
        "candidate_sizing_max_candidates": 8,
        "candidate_sizing_min_mwh": 0.25,
        "candidate_sizing_cap_fraction": 1.0,
        "memory_enabled": memory_enabled,
        "memory_bank_path": MEMORY_BANK if memory_enabled else None,
        "memory_scope_filter": "all",
        "memory_max_items_per_agent": 5,
        "memory_max_prompt_chars": 2400,
        "output_dir": str(RUN_ROOT),
        "llm": {
            "enabled": bool(variant["llm_enabled"]),
            "model": "Qwen/Qwen3-32B",
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "api_key": "heimdall-local",
            "temperature": 0.2,
            "max_tokens": 1000,
            "timeout_seconds": 240,
            "max_concurrency": 12,
            "per_endpoint_max_concurrency": 6,
        },
    }


def _check(full: list[dict[str, Any]], smoke: list[dict[str, Any]]) -> None:
    if len(full) != 12:
        raise RuntimeError(f"expected 12 full configs, got {len(full)}")
    if len(smoke) != 4:
        raise RuntimeError(f"expected 4 smoke configs, got {len(smoke)}")
    run_ids = [payload["run_id"] for payload in [*full, *smoke]]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("duplicate run_id")
    if {payload["chooser_mode"] for payload in full} != {"deterministic_high_fill_accepted", "llm_fill_selector"}:
        raise RuntimeError("unexpected chooser modes")


def _write(full: list[dict[str, Any]], smoke: list[dict[str, Any]]) -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    full_paths = [_write_payload(payload, "full") for payload in full]
    smoke_paths = [_write_payload(payload, "smoke") for payload in smoke]
    (CONFIG_ROOT / "all.txt").write_text("\n".join(str(path) for path in full_paths) + "\n", encoding="utf-8")
    (CONFIG_ROOT / "smoke.txt").write_text("\n".join(str(path) for path in smoke_paths) + "\n", encoding="utf-8")
    (CONFIG_ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "matrix": MATRIX,
                "created_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
                "full_count": len(full_paths),
                "smoke_count": len(smoke_paths),
                "windows": WINDOWS,
                "variants": list(VARIANTS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (CONFIG_ROOT / "RUNBOOK.md").write_text(_runbook(), encoding="utf-8")
    launcher = CONFIG_ROOT / "launch_after_current_matrix.sh"
    launcher.write_text(_launcher(), encoding="utf-8")
    launcher.chmod(0o755)


def _write_payload(payload: dict[str, Any], split: str) -> Path:
    run_id = str(payload["run_id"])
    variant = next(slug for slug in VARIANTS if f"-{slug}-" in run_id)
    path = CONFIG_ROOT / split / variant / f"{payload['run_id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _runbook() -> str:
    return f"""# {MATRIX}

Focused S06 high-fill matrix over apr02-0530, apr09-1830, and apr13-0015.

Run:

```bash
bash {CONFIG_ROOT / "launch_after_current_matrix.sh"}
```
"""


def _launcher() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ucloud/heimdall
export PYTHONPATH=.

uv run python tools/experiments/generate_high_fill_llm_s06_20260522.py --check-only
while read -r cfg; do
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
done < {CONFIG_ROOT}/smoke.txt
while read -r cfg; do
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
done < {CONFIG_ROOT}/all.txt

smoke_log_dir={RUN_ROOT}/logs-smoke-$(date -u +%Y%m%dT%H%M%SZ)
uv run python ai-society/run_long_model_society_matrix.py \\
  --config-list {CONFIG_ROOT}/smoke.txt \\
  --log-dir "$smoke_log_dir" \\
  --skip-vllm-restart

full_log_dir={RUN_ROOT}/logs-$(date -u +%Y%m%dT%H%M%SZ)
tmux new-session -d -s heimdall-high-fill-llm-s06 \\
  "cd /home/ucloud/heimdall && PYTHONPATH=. uv run python ai-society/run_long_model_society_matrix.py --config-list {CONFIG_ROOT}/all.txt --log-dir $full_log_dir --skip-vllm-restart"
echo "launched heimdall-high-fill-llm-s06 log_dir=$full_log_dir"
"""


if __name__ == "__main__":
    raise SystemExit(main())
