from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path("ai-society/configs/intelligence-chair-matrix")
RUN_ROOT = Path("ai-society/runs/intelligence-chair-matrix")
MEMORY_BANK = RUN_ROOT / "memory-v2-bank.jsonl"
CONTEXT_DIR = "data/cache/real_context/april_2026"

WINDOWS = [
    ("apr02-0530", "2026-04-02T05:30:00Z", 24),
    ("apr09-1830", "2026-04-09T18:30:00Z", 24),
    ("apr13-0015", "2026-04-13T00:15:00Z", 24),
]

ARMS = [
    {
        "slug": "8-core",
        "agent_count": 8,
        "profile": "action_core_8",
        "strategy": "comm_broadcast_digest",
        "chooser_mode": "llm",
        "memory": False,
        "max_concurrency": 6,
    },
    {
        "slug": "8-core-chair-2agree",
        "agent_count": 8,
        "profile": "action_core_8",
        "strategy": "comm_society_chair_2agree",
        "chooser_mode": "llm",
        "memory": False,
        "max_concurrency": 6,
    },
    {
        "slug": "8-core-chair-riskveto",
        "agent_count": 8,
        "profile": "action_core_8",
        "strategy": "comm_society_chair_riskveto",
        "chooser_mode": "llm",
        "memory": False,
        "max_concurrency": 6,
    },
    {
        "slug": "8-core-chair-intel",
        "agent_count": 8,
        "profile": "action_core_8",
        "strategy": "comm_society_chair_intel",
        "chooser_mode": "llm",
        "memory": False,
        "max_concurrency": 6,
    },
    {
        "slug": "12-bcast-mem",
        "agent_count": 12,
        "profile": "balanced_intelligence",
        "strategy": "comm_broadcast_digest",
        "chooser_mode": "llm",
        "memory": True,
        "max_concurrency": 8,
    },
    {
        "slug": "det-best-accepted",
        "agent_count": 8,
        "profile": "action_core_8",
        "strategy": "comm_broadcast_digest",
        "chooser_mode": "deterministic_best_accepted",
        "memory": False,
        "max_concurrency": 1,
        "llm_enabled": False,
    },
    {
        "slug": "det-watch-threshold",
        "agent_count": 8,
        "profile": "action_core_8",
        "strategy": "comm_broadcast_digest",
        "chooser_mode": "deterministic_watch_threshold",
        "memory": False,
        "max_concurrency": 1,
        "llm_enabled": False,
    },
]

MEMORY_ARMS = [
    {"slug": "best-chair-no-mem", "memory": False, "memory_scope_filter": "all"},
    {
        "slug": "best-chair-memory-v2-archetype",
        "memory": True,
        "memory_scope_filter": "archetype",
    },
    {
        "slug": "best-chair-memory-v2-chair-only",
        "memory": True,
        "memory_scope_filter": "synthesis",
        "agent_count": 9,
        "profile": "action_core_9_chair",
    },
    {"slug": "best-chair-memory-v2-agent", "memory": True, "memory_scope_filter": "agent"},
]

ROLE_ARMS = [
    {"slug": "action-core-9-chair", "agent_count": 9, "profile": "action_core_9_chair"},
    {"slug": "action-core-10-safety", "agent_count": 10, "profile": "action_core_10_safety"},
    {"slug": "action-core-8-aggressive", "agent_count": 8, "profile": "action_core_8_aggressive"},
    {"slug": "action-core-8-safety", "agent_count": 8, "profile": "action_core_8_safety"},
    {"slug": "action-core-8-toolsplit", "agent_count": 8, "profile": "action_core_8_toolsplit"},
]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    configs: list[Path] = []
    for window_slug, start, ticks in WINDOWS:
        for arm in ARMS:
            configs.append(_write_config("screen", window_slug, start, ticks, arm))
        for arm in MEMORY_ARMS:
            full = {
                "agent_count": 8,
                "profile": "action_core_8",
                "strategy": "comm_society_chair_intel",
                "chooser_mode": "llm",
                "max_concurrency": 6,
                **arm,
            }
            configs.append(_write_config("memory", window_slug, start, ticks, full))
        for arm in ROLE_ARMS:
            full = {
                "strategy": "comm_society_chair_intel",
                "chooser_mode": "llm",
                "memory": False,
                "max_concurrency": 8 if arm["agent_count"] > 8 else 6,
                **arm,
            }
            configs.append(_write_config("roles", window_slug, start, ticks, full))
    gpu0 = [path for idx, path in enumerate(configs) if idx % 2 == 0]
    gpu1 = [path for idx, path in enumerate(configs) if idx % 2 == 1]
    (ROOT / "gpu0.txt").write_text("".join(str(path) + "\n" for path in gpu0), encoding="utf-8")
    (ROOT / "gpu1.txt").write_text("".join(str(path) + "\n" for path in gpu1), encoding="utf-8")
    manifest = {
        "run_count": len(configs),
        "windows": WINDOWS,
        "screen_arms": ARMS,
        "memory_arms": MEMORY_ARMS,
        "role_arms": ROLE_ARMS,
        "memory_bank": str(MEMORY_BANK),
        "promotion_rule": (
            "Start at 24 ticks; expand only arms that preserve explainable "
            "watch-hour quality and reduce bid noise or retain useful profit."
        ),
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_runner()
    print(
        json.dumps(
            {"ok": True, "run_count": len(configs), "manifest": str(ROOT / "manifest.json")},
            indent=2,
        )
    )


def _write_config(
    stage: str,
    window_slug: str,
    start: str,
    ticks: int,
    arm: dict[str, Any],
) -> Path:
    run_id = f"icm-{stage}-{arm['slug']}-{window_slug}-f8-q32"
    path = ROOT / stage / f"{run_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_config(run_id, start, ticks, arm), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _config(run_id: str, start: str, ticks: int, arm: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": 42,
        "zone": "DK1",
        "agent_count": arm["agent_count"],
        "ticks": ticks,
        "start_timestamp": start,
        "forecaster_backend": "f8",
        "chooser_mode": arm["chooser_mode"],
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "objective": "bid_seeking",
        "ablation_strategy": arm["strategy"],
        "persona_profile": arm["profile"],
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
        "memory_enabled": bool(arm.get("memory", False)),
        "memory_bank_path": str(MEMORY_BANK),
        "memory_scope_filter": str(arm.get("memory_scope_filter", "all")),
        "memory_max_items_per_agent": 5,
        "memory_max_prompt_chars": 2400,
        "reviewer_mode": "code_only",
        "llm": {
            "enabled": bool(arm.get("llm_enabled", True)),
            "model": "Qwen/Qwen3-32B",
            "temperature": 0.2,
            "max_tokens": 1000,
            "timeout_seconds": 180,
            "max_concurrency": arm["max_concurrency"],
        },
    }


def _write_runner() -> None:
    script = RUN_ROOT / "run_intelligence_chair_matrix.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /home/ucloud/heimdall",
        "export PYTHONPATH='.'",
        "mkdir -p ai-society/runs/intelligence-chair-matrix/logs",
        f"bank='{MEMORY_BANK}'",
        "touch \"$bank\"",
        "log_dir=\"ai-society/runs/intelligence-chair-matrix/logs/$(date -u +%Y%m%dT%H%M%SZ)\"",
        "mkdir -p \"$log_dir\"",
        "python ai-society/run_market_intelligence_stage.py "
        "--stage intelligence-chair --gpu gpu0 "
        "--base-url http://127.0.0.1:8000/v1 "
        "--config-list ai-society/configs/intelligence-chair-matrix/gpu0.txt "
        "--log-dir \"$log_dir\" > \"$log_dir/gpu0.stdout.log\" 2>&1 &",
        "pid0=$!",
        "python ai-society/run_market_intelligence_stage.py "
        "--stage intelligence-chair --gpu gpu1 "
        "--base-url http://127.0.0.1:8001/v1 "
        "--config-list ai-society/configs/intelligence-chair-matrix/gpu1.txt "
        "--log-dir \"$log_dir\" > \"$log_dir/gpu1.stdout.log\" 2>&1 &",
        "pid1=$!",
        "echo \"$pid0 $pid1\" > \"$log_dir/pids.txt\"",
        "echo \"launched intelligence chair matrix log_dir=$log_dir gpu0=$pid0 gpu1=$pid1\"",
        "wait",
    ]
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)


if __name__ == "__main__":
    main()
