from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/overnight-broadcast-ablation")
RUN_ROOT = Path("ai-society/runs/overnight-broadcast-ablation")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"

WINDOWS = [
    {
        "slug": "apr02-0530",
        "start": "2026-04-02T05:30:00Z",
        "mixed20_run_id": "msa-screen-mixed20-apr02-0530-24",
        "mixed20_run_dir": "ai-society/runs/mixed-sideaware-20260515/msa-screen-mixed20-apr02-0530-24",
    },
    {
        "slug": "apr03-1430",
        "start": "2026-04-03T14:30:00Z",
        "mixed20_run_id": "msa-screen-mixed20-apr03-1430-24",
        "mixed20_run_dir": "ai-society/runs/mixed-sideaware-20260515/msa-screen-mixed20-apr03-1430-24",
    },
    {
        "slug": "apr05-1030",
        "start": "2026-04-05T10:30:00Z",
        "mixed20_run_id": "msa-screen-mixed20-apr05-1030-24",
        "mixed20_run_dir": "ai-society/runs/mixed-sideaware-20260515/msa-screen-mixed20-apr05-1030-24",
    },
    {
        "slug": "apr09-1830",
        "start": "2026-04-09T18:30:00Z",
        "mixed20_run_id": "msa-screen-mixed20-apr09-1830-24",
        "mixed20_run_dir": "ai-society/runs/mixed-sideaware-20260515/msa-screen-mixed20-apr09-1830-24",
    },
    {
        "slug": "apr13-0015",
        "start": "2026-04-13T00:15:00Z",
        "mixed20_run_id": "msa-screen-mixed20-apr13-0015-24",
        "mixed20_run_dir": "ai-society/runs/mixed-sideaware-20260515/msa-screen-mixed20-apr13-0015-24",
    },
]
MODELS = [
    ("q14", "Qwen/Qwen3-14B"),
    ("q32", "Qwen/Qwen3-32B"),
]
SOCIETIES = [
    {"slug": "s06-actioncore", "agent_count": 6, "profile": "action_core_8", "max_concurrency": 8, "per_endpoint_max_concurrency": 4},
    {"slug": "s12-balanced", "agent_count": 12, "profile": "balanced_intelligence", "max_concurrency": 12, "per_endpoint_max_concurrency": 6},
]
SEED = 42


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    _assert_mixed20_references()

    configs: list[Path] = []
    for society in SOCIETIES:
        for model_slug, model in MODELS:
            for window in WINDOWS:
                run_id = f"oba-{society['slug']}-bcast-{window['slug']}-seed{SEED}-{model_slug}"
                path = ROOT / society["slug"] / model_slug / f"{run_id}.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    yaml.safe_dump(_config(run_id, window["start"], model, society), sort_keys=False),
                    encoding="utf-8",
                )
                configs.append(path)

    (ROOT / "config-list.txt").write_text("".join(str(path) + "\n" for path in configs), encoding="utf-8")
    manifest = {
        "run_count": len(configs),
        "seed": SEED,
        "ticks": 24,
        "communication_arm": {"slug": "bcast", "strategy": "comm_broadcast_digest"},
        "windows": WINDOWS,
        "models": MODELS,
        "societies": SOCIETIES,
        "existing_mixed20_baselines": [
            {
                "run_id": window["mixed20_run_id"],
                "run_dir": window["mixed20_run_dir"],
                "model": "Qwen/Qwen3-32B",
                "agent_count": 20,
                "persona_profile": "mixed_expert_20_sideaware",
                "strategy": "comm_broadcast_digest",
                "window": window["slug"],
            }
            for window in WINDOWS
        ],
        "configs": [str(path) for path in configs],
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "comparison-baselines.json").write_text(
        json.dumps(manifest["existing_mixed20_baselines"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_runner()
    print(json.dumps({"ok": True, "run_count": len(configs), "manifest": str(ROOT / "manifest.json")}, indent=2))


def _config(run_id: str, start: str, model: str, society: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": SEED,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": society["agent_count"],
        "ticks": 24,
        "start_timestamp": start,
        "forecaster_backend": "f8",
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "objective": "bid_seeking",
        "ablation_strategy": "comm_broadcast_digest",
        "persona_profile": society["profile"],
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "p2h_only_simulator",
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
            "model": model,
            "temperature": 0.2,
            "max_tokens": 512,
            "timeout_seconds": 180,
            "max_concurrency": society["max_concurrency"],
            "per_endpoint_max_concurrency": society["per_endpoint_max_concurrency"],
        },
    }


def _assert_mixed20_references() -> None:
    missing = [window["mixed20_run_dir"] for window in WINDOWS if not Path(window["mixed20_run_dir"]).joinpath("summary.json").exists()]
    if missing:
        raise RuntimeError(f"missing mixed-20 baseline summaries: {missing}")


def _write_runner() -> None:
    script = RUN_ROOT / "run_overnight_broadcast_ablation.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /home/ucloud/heimdall",
        "export PYTHONPATH='.'",
        "mkdir -p ai-society/runs/overnight-broadcast-ablation/logs",
        "log_dir=\"ai-society/runs/overnight-broadcast-ablation/logs/$(date -u +%Y%m%dT%H%M%SZ)\"",
        "mkdir -p \"$log_dir\"",
        "echo \"[$(date -Is)] overnight broadcast ablation start log_dir=$log_dir\"",
        "uv run python ai-society/run_long_model_society_matrix.py \\",
        "  --config-list ai-society/configs/overnight-broadcast-ablation/config-list.txt \\",
        "  --log-dir \"$log_dir\" \\",
        "  --continue-on-failure \\",
        "  > \"$log_dir/controller.stdout.log\" 2>&1",
        "echo \"[$(date -Is)] overnight broadcast ablation complete log_dir=$log_dir\"",
    ]
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)


if __name__ == "__main__":
    main()
