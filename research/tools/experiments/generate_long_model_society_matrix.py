from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/long-model-society-matrix")
RUN_ROOT = Path("ai-society/runs/long-model-society-matrix")
CONTEXT_DIR = "data/cache/real_context/april_2026"

WINDOWS = [
    ("apr02-0530", "2026-04-02T05:30:00Z"),
    ("apr03-1430", "2026-04-03T14:30:00Z"),
    ("apr07-1715", "2026-04-07T17:15:00Z"),
    ("apr09-1830", "2026-04-09T18:30:00Z"),
    ("apr13-0700", "2026-04-13T07:00:00Z"),
    ("apr22-0830", "2026-04-22T08:30:00Z"),
    ("apr27-1730", "2026-04-27T17:30:00Z"),
]
MODELS = [
    ("q8", "Qwen/Qwen3-8B"),
    ("q14", "Qwen/Qwen3-14B"),
    ("q32", "Qwen/Qwen3-32B"),
]
SEEDS = [13, 42, 137]
SOCIETIES = [
    {"slug": "s06-actioncore", "agent_count": 6, "profile": "action_core_8", "max_concurrency": 8, "per_endpoint_max_concurrency": 4},
    {"slug": "s12-balanced", "agent_count": 12, "profile": "balanced_intelligence", "max_concurrency": 12, "per_endpoint_max_concurrency": 6},
    {"slug": "s20-mixed", "agent_count": 20, "profile": "mixed_expert_20_sideaware", "max_concurrency": 16, "per_endpoint_max_concurrency": 8},
]
COMM_ARMS = [
    {"slug": "bcast", "strategy": "comm_broadcast_digest"},
    {"slug": "chair", "strategy": "comm_society_chair_intel"},
    {"slug": "ind", "strategy": "diverse_action_society"},
]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    configs: list[Path] = []
    for society in SOCIETIES:
        for model_slug, model in MODELS:
            for window_slug, start in WINDOWS:
                for comm in COMM_ARMS:
                    for seed in SEEDS:
                        run_id = f"lmsm-{society['slug']}-{comm['slug']}-{window_slug}-seed{seed}-{model_slug}"
                        path = ROOT / society["slug"] / model_slug / f"{run_id}.yaml"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(
                            yaml.safe_dump(_config(run_id, start, seed, model, society, comm), sort_keys=False),
                            encoding="utf-8",
                        )
                        configs.append(path)
    (ROOT / "config-list.txt").write_text("".join(str(path) + "\n" for path in configs), encoding="utf-8")
    manifest = {
        "run_count": len(configs),
        "windows": WINDOWS,
        "models": MODELS,
        "seeds": SEEDS,
        "societies": SOCIETIES,
        "communication_arms": COMM_ARMS,
        "ticks": 24,
        "configs": [str(path) for path in configs],
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_runner()
    print(json.dumps({"ok": True, "run_count": len(configs), "manifest": str(ROOT / "manifest.json")}, indent=2))


def _config(run_id: str, start: str, seed: int, model: str, society: dict[str, Any], comm: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": seed,
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
        "ablation_strategy": comm["strategy"],
        "persona_profile": society["profile"],
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "p2h_only_simulator",
        "max_tool_rounds": 6,
        "data_start": "2026-04-01T00:00:00Z",
        "data_end": "2026-05-01T00:00:00Z",
        "context_dataset_dir": CONTEXT_DIR,
        "data_cache_dir": f"{CONTEXT_DIR}/source_cache",
        "default_lookback_hours": 24,
        "cache_refresh": False,
        "output_dir": str(RUN_ROOT),
        "reviewer_mode": "code_only",
        "llm": {
            "enabled": True,
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "api_key": "heimdall-local",
            "model": model,
            "temperature": 0.2,
            "max_tokens": 1000,
            "timeout_seconds": 180,
            "max_concurrency": society["max_concurrency"],
            "per_endpoint_max_concurrency": society["per_endpoint_max_concurrency"],
        },
    }


def _write_runner() -> None:
    script = RUN_ROOT / "run_long_model_society_matrix.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /home/ucloud/heimdall",
        "export PYTHONPATH='.'",
        "mkdir -p ai-society/runs/long-model-society-matrix/logs",
        "log_dir=\"ai-society/runs/long-model-society-matrix/logs/$(date -u +%Y%m%dT%H%M%SZ)\"",
        "mkdir -p \"$log_dir\"",
        "echo \"[$(date -Is)] long model society matrix start log_dir=$log_dir\"",
        "uv run python ai-society/run_long_model_society_matrix.py \\",
        "  --config-list ai-society/configs/long-model-society-matrix/config-list.txt \\",
        "  --log-dir \"$log_dir\" \\",
        "  --continue-on-failure \\",
        "  > \"$log_dir/controller.stdout.log\" 2>&1",
        "echo \"[$(date -Is)] long model society matrix complete log_dir=$log_dir\"",
    ]
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)


if __name__ == "__main__":
    main()
