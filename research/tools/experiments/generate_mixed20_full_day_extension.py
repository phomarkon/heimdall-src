from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/mixed20-full-day-extension")
RUN_ROOT = Path("ai-society/runs/mixed20-full-day-extension")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"
MODEL = "Qwen/Qwen3-32B"

DAYS = {
    "apr08": "2026-04-08T00:00:00Z",
    "apr09": "2026-04-09T00:00:00Z",
}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    configs: list[Path] = []
    seen: set[str] = set()
    for day_slug in DAYS:
        run_id = f"m20fde-mixed20-{day_slug}-96-real-controls-q32"
        path = ROOT / f"{run_id}.yaml"
        _write_config(path, _config(run_id, day_slug), seen)
        configs.append(path)
    _assert_no_existing_outputs(seen)
    (ROOT / "config-list.txt").write_text("".join(f"{path}\n" for path in configs), encoding="utf-8")
    (ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "run_count": len(configs),
                "model": MODEL,
                "ticks": 96,
                "config_list": str(ROOT / "config-list.txt"),
                "run_root": str(RUN_ROOT),
                "runs": [{"run_id": path.stem, "config": str(path)} for path in configs],
                "continues_real_control_full_days_after": "fob-mixed20-apr07-96-real-controls-q32",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_runner()
    print(json.dumps({"ok": True, "run_count": len(configs), "config_list": str(ROOT / "config-list.txt")}, indent=2))


def _write_config(path: Path, payload: dict[str, Any], seen: set[str]) -> None:
    run_id = str(payload["run_id"])
    if run_id in seen:
        raise RuntimeError(f"duplicate run_id: {run_id}")
    seen.add(run_id)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _config(run_id: str, day_slug: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": 42,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": 20,
        "ticks": 96,
        "start_timestamp": DAYS[day_slug],
        "forecaster_backend": "f8",
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "objective": "bid_seeking",
        "ablation_strategy": "comm_broadcast_digest",
        "persona_profile": "mixed_expert_20_sideaware",
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "asset_simulator_v1",
        "asset_simulator_mode": "dual_compare_real_controls",
        "asset_proxy_style": "market",
        "max_tool_rounds": 6,
        "simulator_max_concurrency": 8,
        "data_start": "2026-04-01T00:00:00Z",
        "data_end": "2026-05-01T00:00:00Z",
        "context_dataset_dir": CONTEXT_DIR,
        "data_cache_dir": f"{CONTEXT_DIR}/source_cache",
        "default_lookback_hours": 24,
        "cache_refresh": False,
        "output_dir": str(RUN_ROOT),
        "memory_enabled": True,
        "memory_bank_path": MEMORY_BANK,
        "memory_max_items_per_agent": 5,
        "memory_max_prompt_chars": 2400,
        "reviewer_mode": "code_only",
        "llm": {
            "enabled": True,
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "api_key": "heimdall-local",
            "model": MODEL,
            "temperature": 0.2,
            "max_tokens": 512,
            "timeout_seconds": 180,
            "max_concurrency": 12,
            "per_endpoint_max_concurrency": 6,
        },
    }


def _assert_no_existing_outputs(run_ids: set[str]) -> None:
    collisions: list[str] = []
    for run_id in sorted(run_ids):
        candidates = [RUN_ROOT / run_id, Path("evaluations") / run_id]
        collisions.extend(str(path) for path in candidates if path.exists())
    if collisions:
        raise RuntimeError("refusing to generate duplicate full-day extension outputs:\n" + "\n".join(collisions))


def _write_runner() -> None:
    script = RUN_ROOT / "run_mixed20_full_day_extension.sh"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd /home/ucloud/heimdall",
                "export PYTHONPATH='.'",
                "mkdir -p ai-society/runs/mixed20-full-day-extension/logs",
                "log_dir=\"ai-society/runs/mixed20-full-day-extension/logs/$(date -u +%Y%m%dT%H%M%SZ)\"",
                "mkdir -p \"$log_dir\"",
                "echo \"[$(date -Is)] mixed20 full-day extension start log_dir=$log_dir\"",
                "uv run python ai-society/run_long_model_society_matrix.py \\",
                "  --config-list ai-society/configs/mixed20-full-day-extension/config-list.txt \\",
                "  --log-dir \"$log_dir\" \\",
                "  --continue-on-failure \\",
                "  > \"$log_dir/controller.stdout.log\" 2>&1",
                "echo \"[$(date -Is)] mixed20 full-day extension complete log_dir=$log_dir\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


if __name__ == "__main__":
    main()
