from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/final-core-overnight-matrix")
RUN_ROOT = Path("ai-society/runs/final-core-overnight-matrix")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"
MODEL = "Qwen/Qwen3-32B"

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr07-1715": "2026-04-07T17:15:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}

SOCIETIES = {
    "s06-actioncore": {
        "agent_count": 6,
        "persona_profile": "action_core_8",
        "max_concurrency": 8,
        "per_endpoint_max_concurrency": 4,
        "max_tokens": 1000,
        "memory": False,
    },
    "s12-balanced": {
        "agent_count": 12,
        "persona_profile": "balanced_intelligence",
        "max_concurrency": 12,
        "per_endpoint_max_concurrency": 6,
        "max_tokens": 512,
        "memory": True,
    },
    "s20-mixed": {
        "agent_count": 20,
        "persona_profile": "mixed_expert_20_sideaware",
        "max_concurrency": 16,
        "per_endpoint_max_concurrency": 8,
        "max_tokens": 1000,
        "memory": False,
    },
}

RUNS = [
    # Fill apr02 seed robustness for the smaller and medium societies.
    ("s06-actioncore", "apr02-0530", 13, "core_seed_fill"),
    ("s06-actioncore", "apr02-0530", 137, "core_seed_fill"),
    ("s12-balanced", "apr02-0530", 13, "core_seed_fill"),
    ("s12-balanced", "apr02-0530", 137, "core_seed_fill"),
    # Fill s20 seed robustness across the three core windows.
    ("s20-mixed", "apr02-0530", 13, "s20_core_seed_fill"),
    ("s20-mixed", "apr02-0530", 137, "s20_core_seed_fill"),
    ("s20-mixed", "apr09-1830", 13, "s20_core_seed_fill"),
    ("s20-mixed", "apr09-1830", 137, "s20_core_seed_fill"),
    ("s20-mixed", "apr13-0015", 13, "s20_core_seed_fill"),
    ("s20-mixed", "apr13-0015", 42, "s20_core_seed_fill"),
    ("s20-mixed", "apr13-0015", 137, "s20_core_seed_fill"),
    # Sparse control fill.
    ("s06-actioncore", "apr07-1715", 42, "sparse_control_fill"),
    ("s20-mixed", "apr07-1715", 42, "sparse_control_fill"),
]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    configs: list[Path] = []
    seen: set[str] = set()
    for society_slug, window_slug, seed, purpose in RUNS:
        run_id = f"fco-{society_slug}-bcast-{window_slug}-seed{seed}-q32"
        path = ROOT / purpose / society_slug / f"{run_id}.yaml"
        _write_config(path, _config(run_id, society_slug, window_slug, seed), seen)
        configs.append(path)

    _assert_no_existing_outputs(seen)
    (ROOT / "config-list.txt").write_text("".join(f"{path}\n" for path in configs), encoding="utf-8")
    _write_manifest(configs)
    _write_runner()
    print(
        json.dumps(
            {"ok": True, "run_count": len(configs), "config_list": str(ROOT / "config-list.txt")},
            indent=2,
        )
    )


def _write_config(path: Path, payload: dict[str, Any], seen: set[str]) -> None:
    run_id = str(payload["run_id"])
    if run_id in seen:
        raise RuntimeError(f"duplicate run_id: {run_id}")
    seen.add(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _config(run_id: str, society_slug: str, window_slug: str, seed: int) -> dict[str, Any]:
    society = SOCIETIES[society_slug]
    payload: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": society["agent_count"],
        "ticks": 24,
        "start_timestamp": WINDOWS[window_slug],
        "forecaster_backend": "f8",
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "objective": "bid_seeking",
        "ablation_strategy": "comm_broadcast_digest",
        "persona_profile": society["persona_profile"],
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
            "model": MODEL,
            "temperature": 0.2,
            "max_tokens": society["max_tokens"],
            "timeout_seconds": 180,
            "max_concurrency": society["max_concurrency"],
            "per_endpoint_max_concurrency": society["per_endpoint_max_concurrency"],
        },
    }
    if society["memory"]:
        payload.update(
            {
                "simulator_max_concurrency": 8,
                "memory_enabled": True,
                "memory_bank_path": MEMORY_BANK,
                "memory_max_items_per_agent": 5,
                "memory_max_prompt_chars": 2400,
            }
        )
    return payload


def _assert_no_existing_outputs(run_ids: set[str]) -> None:
    collisions: list[str] = []
    for run_id in sorted(run_ids):
        candidates = [RUN_ROOT / run_id, Path("evaluations") / run_id]
        collisions.extend(str(path) for path in candidates if path.exists())
    if collisions:
        raise RuntimeError("refusing to generate duplicate final-core overnight outputs:\n" + "\n".join(collisions))


def _write_manifest(configs: list[Path]) -> None:
    payload = {
        "run_count": len(configs),
        "model": MODEL,
        "forecaster_backend": "f8",
        "forecaster_seed": 42,
        "ablation_strategy": "comm_broadcast_digest",
        "config_list": str(ROOT / "config-list.txt"),
        "run_root": str(RUN_ROOT),
        "runs": [
            {
                "run_id": path.stem,
                "config": str(path),
                "purpose": path.parts[len(ROOT.parts)],
            }
            for path in configs
        ],
        "known_existing_inputs_not_duplicated": [
            "s06/s12/s20 q32 seed42 core cells where present",
            "s06/s12 q32 apr09/apr13 seed13/seed137 final-completion cells",
            "s06/s20 q32 apr06/apr22/apr27 sparse controls",
        ],
    }
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_runner() -> None:
    script = RUN_ROOT / "run_final_core_overnight_matrix.sh"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd /home/ucloud/heimdall",
                "export PYTHONPATH='.'",
                "mkdir -p ai-society/runs/final-core-overnight-matrix/logs",
                "log_dir=\"ai-society/runs/final-core-overnight-matrix/logs/$(date -u +%Y%m%dT%H%M%SZ)\"",
                "mkdir -p \"$log_dir\"",
                "echo \"[$(date -Is)] final core overnight matrix start log_dir=$log_dir\"",
                "uv run python ai-society/run_long_model_society_matrix.py \\",
                "  --config-list ai-society/configs/final-core-overnight-matrix/config-list.txt \\",
                "  --log-dir \"$log_dir\" \\",
                "  --continue-on-failure \\",
                "  > \"$log_dir/controller.stdout.log\" 2>&1",
                "echo \"[$(date -Is)] final core overnight matrix complete log_dir=$log_dir\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


if __name__ == "__main__":
    main()
