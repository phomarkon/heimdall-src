from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/final-completion-matrix")
RUN_ROOT = Path("ai-society/runs/final-completion-matrix")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr06-1300": "2026-04-06T13:00:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
    "apr22-0830": "2026-04-22T08:30:00Z",
    "apr27-1730": "2026-04-27T17:30:00Z",
}

MODELS = {
    "q14": "Qwen/Qwen3-14B",
    "q32": "Qwen/Qwen3-32B",
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
    # Q32 seed robustness for the core s12/s06 thesis comparison.
    ("s12-balanced", "q32", "apr09-1830", 13, "seed_robustness"),
    ("s12-balanced", "q32", "apr13-0015", 13, "seed_robustness"),
    ("s12-balanced", "q32", "apr09-1830", 137, "seed_robustness"),
    ("s12-balanced", "q32", "apr13-0015", 137, "seed_robustness"),
    ("s06-actioncore", "q32", "apr09-1830", 13, "seed_robustness"),
    ("s06-actioncore", "q32", "apr13-0015", 13, "seed_robustness"),
    ("s06-actioncore", "q32", "apr09-1830", 137, "seed_robustness"),
    ("s06-actioncore", "q32", "apr13-0015", 137, "seed_robustness"),
    # Q14 fill for s20 model-scale comparison.
    ("s20-mixed", "q14", "apr02-0530", 42, "q14_core_fill"),
    ("s20-mixed", "q14", "apr09-1830", 42, "q14_core_fill"),
    # Q14 fill for the breadth windows.
    ("s06-actioncore", "q14", "apr06-1300", 42, "q14_breadth_fill"),
    ("s06-actioncore", "q14", "apr22-0830", 42, "q14_breadth_fill"),
    ("s06-actioncore", "q14", "apr27-1730", 42, "q14_breadth_fill"),
    ("s20-mixed", "q14", "apr06-1300", 42, "q14_breadth_fill"),
    ("s20-mixed", "q14", "apr22-0830", 42, "q14_breadth_fill"),
    ("s20-mixed", "q14", "apr27-1730", 42, "q14_breadth_fill"),
]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    configs: list[Path] = []
    seen: set[str] = set()
    for society_slug, model_slug, window_slug, seed, purpose in RUNS:
        run_id = f"fcm-{society_slug}-bcast-{window_slug}-seed{seed}-{model_slug}"
        path = ROOT / purpose / society_slug / model_slug / f"{run_id}.yaml"
        _write_config(path, _config(run_id, society_slug, model_slug, window_slug, seed), seen)
        configs.append(path)

    _assert_no_existing_outputs(seen)
    (ROOT / "config-list.txt").write_text("".join(f"{path}\n" for path in configs), encoding="utf-8")
    _write_manifest(configs)
    _write_runner()

    print(
        json.dumps(
            {
                "ok": True,
                "run_count": len(configs),
                "config_list": str(ROOT / "config-list.txt"),
                "run_root": str(RUN_ROOT),
            },
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


def _config(run_id: str, society_slug: str, model_slug: str, window_slug: str, seed: int) -> dict[str, Any]:
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
            "model": MODELS[model_slug],
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
        candidates = [
            RUN_ROOT / run_id,
            Path("evaluations") / run_id,
        ]
        collisions.extend(str(path) for path in candidates if path.exists())
    if collisions:
        raise RuntimeError("refusing to generate duplicate final-completion outputs:\n" + "\n".join(collisions))


def _write_manifest(configs: list[Path]) -> None:
    payload = {
        "run_count": len(configs),
        "models": MODELS,
        "forecaster_backend": "f8",
        "ablation_strategy": "comm_broadcast_digest",
        "config_list": str(ROOT / "config-list.txt"),
        "run_root": str(RUN_ROOT),
        "runs": [
            {
                "run_id": path.stem,
                "config": str(path),
                "model_slug": "q14" if path.stem.endswith("-q14") else "q32",
                "purpose": path.parts[len(ROOT.parts)],
            }
            for path in configs
        ],
        "known_existing_inputs_not_duplicated": [
            "oba-s06-actioncore-bcast-*-seed42-q14",
            "oba-s12-balanced-bcast-*-seed42-q14",
            "foa/fob/oba q32 seed42 cells",
            "lmsm-s06-actioncore-bcast-apr02-0530-seed{13,42,137}-q8",
        ],
    }
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_runner() -> None:
    script = RUN_ROOT / "run_final_completion_matrix.sh"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd /home/ucloud/heimdall",
                "export PYTHONPATH='.'",
                "mkdir -p ai-society/runs/final-completion-matrix/logs",
                "log_dir=\"ai-society/runs/final-completion-matrix/logs/$(date -u +%Y%m%dT%H%M%SZ)\"",
                "mkdir -p \"$log_dir\"",
                "echo \"[$(date -Is)] final completion matrix start log_dir=$log_dir\"",
                "uv run python ai-society/run_long_model_society_matrix.py \\",
                "  --config-list ai-society/configs/final-completion-matrix/config-list.txt \\",
                "  --log-dir \"$log_dir\" \\",
                "  --continue-on-failure \\",
                "  > \"$log_dir/controller.stdout.log\" 2>&1",
                "echo \"[$(date -Is)] final completion matrix complete log_dir=$log_dir\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


if __name__ == "__main__":
    main()
