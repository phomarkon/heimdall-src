from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/final-option-a-model-scale")
RUN_ROOT = Path("ai-society/runs/final-option-a-model-scale")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}

MODELS = {
    "q8": "Qwen/Qwen3-8B",
    "q32": "Qwen/Qwen3-32B",
    "q72": "Qwen/Qwen2.5-72B-Instruct",
}

SOCIETIES = {
    "s06-actioncore": {
        "agent_count": 6,
        "profile": "action_core_8",
        "max_concurrency": 8,
        "per_endpoint_max_concurrency": 4,
        "max_tokens": 1000,
        "memory": False,
    },
    "s12-balanced": {
        "agent_count": 12,
        "profile": "balanced_intelligence",
        "max_concurrency": 12,
        "per_endpoint_max_concurrency": 6,
        "max_tokens": 512,
        "memory": True,
    },
    "s20-mixed": {
        "agent_count": 20,
        "profile": "mixed_expert_20_sideaware",
        "max_concurrency": 16,
        "per_endpoint_max_concurrency": 8,
        "max_tokens": 1000,
        "memory": False,
    },
}

RUNS = [
    # Priority 1: s12 8B, then s12 72B.
    ("s12-balanced", "q8", "apr02-0530"),
    ("s12-balanced", "q8", "apr09-1830"),
    ("s12-balanced", "q8", "apr13-0015"),
    ("s12-balanced", "q72", "apr02-0530"),
    ("s12-balanced", "q72", "apr09-1830"),
    ("s12-balanced", "q72", "apr13-0015"),
    # Priority 2: s06 gaps.
    ("s06-actioncore", "q8", "apr09-1830"),
    ("s06-actioncore", "q8", "apr13-0015"),
    ("s06-actioncore", "q72", "apr02-0530"),
    ("s06-actioncore", "q72", "apr09-1830"),
    ("s06-actioncore", "q72", "apr13-0015"),
    # Priority 3: small s20 LLM check.
    ("s20-mixed", "q32", "apr02-0530"),
    ("s20-mixed", "q32", "apr09-1830"),
    ("s20-mixed", "q72", "apr02-0530"),
    ("s20-mixed", "q72", "apr09-1830"),
]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    configs: list[Path] = []
    seen: set[str] = set()
    for society_slug, model_slug, window_slug in RUNS:
        run_id = f"foa-{society_slug}-bcast-{window_slug}-seed42-{model_slug}"
        if run_id in seen:
            raise RuntimeError(f"duplicate run_id: {run_id}")
        seen.add(run_id)
        path = ROOT / society_slug / model_slug / f"{run_id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(_config(run_id, society_slug, model_slug, window_slug), sort_keys=False),
            encoding="utf-8",
        )
        configs.append(path)

    (ROOT / "config-list.txt").write_text("".join(f"{path}\n" for path in configs), encoding="utf-8")
    (ROOT / "resume-from-run4-config-list.txt").write_text(
        "".join(f"{path}\n" for path in configs[3:]),
        encoding="utf-8",
    )
    (ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "run_count": len(configs),
                "seed": 42,
                "ticks": 24,
                "forecaster_backend": "f8",
                "ablation_strategy": "comm_broadcast_digest",
                "runs": [
                    {
                        "run_id": path.stem,
                        "config": str(path),
                        "priority": _priority(path.stem),
                    }
                    for path in configs
                ],
                "skipped_as_existing": [
                    "oba-s06-actioncore-bcast-*-seed42-q32",
                    "oba-s12-balanced-bcast-*-seed42-q32",
                    "lmsm-s06-actioncore-bcast-apr02-0530-seed42-q8",
                ],
                "resume_from_run4_config_list": str(ROOT / "resume-from-run4-config-list.txt"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_runner()
    print(json.dumps({"ok": True, "run_count": len(configs), "config_list": str(ROOT / "config-list.txt")}, indent=2))


def _config(run_id: str, society_slug: str, model_slug: str, window_slug: str) -> dict[str, Any]:
    society = SOCIETIES[society_slug]
    config: dict[str, Any] = {
        "run_id": run_id,
        "seed": 42,
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
            "model": MODELS[model_slug],
            "temperature": 0.2,
            "max_tokens": society["max_tokens"],
            "timeout_seconds": 180,
            "max_concurrency": society["max_concurrency"],
            "per_endpoint_max_concurrency": society["per_endpoint_max_concurrency"],
        },
    }
    if society["memory"]:
        config.update(
            {
                "simulator_max_concurrency": 8,
                "memory_enabled": True,
                "memory_bank_path": MEMORY_BANK,
                "memory_max_items_per_agent": 5,
                "memory_max_prompt_chars": 2400,
            }
        )
    return config


def _priority(run_id: str) -> int:
    if "s12-balanced" in run_id:
        return 1
    if "s06-actioncore" in run_id:
        return 2
    return 3


def _write_runner() -> None:
    script = RUN_ROOT / "run_final_option_a_model_scale.sh"
    resume_script = RUN_ROOT / "resume_final_option_a_model_scale_from_run4.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /home/ucloud/heimdall",
        "export PYTHONPATH='.'",
        "mkdir -p ai-society/runs/final-option-a-model-scale/logs",
        "log_dir=\"ai-society/runs/final-option-a-model-scale/logs/$(date -u +%Y%m%dT%H%M%SZ)\"",
        "mkdir -p \"$log_dir\"",
        "echo \"[$(date -Is)] final option A model-scale matrix start log_dir=$log_dir\"",
        "uv run python ai-society/run_long_model_society_matrix.py \\",
        "  --config-list ai-society/configs/final-option-a-model-scale/config-list.txt \\",
        "  --log-dir \"$log_dir\" \\",
        "  --continue-on-failure \\",
        "  > \"$log_dir/controller.stdout.log\" 2>&1",
        "echo \"[$(date -Is)] final option A model-scale matrix complete log_dir=$log_dir\"",
    ]
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)
    resume_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /home/ucloud/heimdall",
        "export PYTHONPATH='.'",
        "mkdir -p ai-society/runs/final-option-a-model-scale/logs",
        "log_dir=\"ai-society/runs/final-option-a-model-scale/logs/$(date -u +%Y%m%dT%H%M%SZ)-resume-run4\"",
        "mkdir -p \"$log_dir\"",
        "echo \"[$(date -Is)] final option A model-scale resume from run 4 start log_dir=$log_dir\"",
        "uv run python ai-society/run_long_model_society_matrix.py \\",
        "  --config-list ai-society/configs/final-option-a-model-scale/resume-from-run4-config-list.txt \\",
        "  --log-dir \"$log_dir\" \\",
        "  --continue-on-failure \\",
        "  > \"$log_dir/controller.stdout.log\" 2>&1",
        "echo \"[$(date -Is)] final option A model-scale resume from run 4 complete log_dir=$log_dir\"",
    ]
    resume_script.write_text("\n".join(resume_lines) + "\n", encoding="utf-8")
    resume_script.chmod(0o755)


if __name__ == "__main__":
    main()
