from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/overnight-broadcast-ablation/deterministic-baselines")
RUN_ROOT = Path("ai-society/runs/overnight-broadcast-ablation-deterministic")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"

WINDOWS = [
    ("apr02-0530", "2026-04-02T05:30:00Z"),
    ("apr03-1430", "2026-04-03T14:30:00Z"),
    ("apr05-1030", "2026-04-05T10:30:00Z"),
    ("apr09-1830", "2026-04-09T18:30:00Z"),
    ("apr13-0015", "2026-04-13T00:15:00Z"),
]
SOCIETIES = [
    {"slug": "s06-actioncore", "agent_count": 6, "profile": "action_core_8"},
    {"slug": "s12-balanced", "agent_count": 12, "profile": "balanced_intelligence"},
    {"slug": "s20-mixed", "agent_count": 20, "profile": "mixed_expert_20_sideaware"},
]
SEED = 42


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    configs: list[Path] = []
    for society in SOCIETIES:
        for window_slug, start in WINDOWS:
            run_id = f"oba-det-{society['slug']}-bcast-{window_slug}-seed{SEED}"
            path = ROOT / society["slug"] / f"{run_id}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(_config(run_id, start, society), sort_keys=False), encoding="utf-8")
            configs.append(path)

    (ROOT / "config-list.txt").write_text("".join(str(path) + "\n" for path in configs), encoding="utf-8")
    manifest = {
        "run_count": len(configs),
        "seed": SEED,
        "ticks": 24,
        "chooser_mode": "deterministic_best_accepted",
        "communication_arm": {"slug": "bcast", "strategy": "comm_broadcast_digest"},
        "windows": WINDOWS,
        "societies": SOCIETIES,
        "configs": [str(path) for path in configs],
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_runner()
    _write_chain_runner()
    print(json.dumps({"ok": True, "run_count": len(configs), "manifest": str(ROOT / "manifest.json")}, indent=2))


def _config(run_id: str, start: str, society: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": SEED,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": society["agent_count"],
        "ticks": 24,
        "start_timestamp": start,
        "forecaster_backend": "f8",
        "chooser_mode": "deterministic_best_accepted",
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
            "enabled": False,
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "api_key": "heimdall-local",
            "model": "deterministic_best_accepted",
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout_seconds": 180,
            "max_concurrency": 1,
            "per_endpoint_max_concurrency": 1,
        },
    }


def _write_runner() -> None:
    script = RUN_ROOT / "run_overnight_deterministic_baselines.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /home/ucloud/heimdall",
        "export PYTHONPATH='.'",
        "mkdir -p ai-society/runs/overnight-broadcast-ablation-deterministic/logs",
        "log_dir=\"ai-society/runs/overnight-broadcast-ablation-deterministic/logs/$(date -u +%Y%m%dT%H%M%SZ)\"",
        "mkdir -p \"$log_dir\"",
        "echo \"[$(date -Is)] overnight deterministic baselines start log_dir=$log_dir\"",
        "uv run python ai-society/run_overnight_deterministic_baselines.py \\",
        "  --config-list ai-society/configs/overnight-broadcast-ablation/deterministic-baselines/config-list.txt \\",
        "  --log-dir \"$log_dir\" \\",
        "  --continue-on-failure \\",
        "  > \"$log_dir/controller.stdout.log\" 2>&1",
        "echo \"[$(date -Is)] overnight deterministic baselines complete log_dir=$log_dir\"",
    ]
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)


def _write_chain_runner() -> None:
    script = RUN_ROOT / "chain_after_overnight_broadcast.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /home/ucloud/heimdall",
        "mkdir -p ai-society/runs/overnight-broadcast-ablation-deterministic/logs",
        "chain_log=\"ai-society/runs/overnight-broadcast-ablation-deterministic/logs/chain-$(date -u +%Y%m%dT%H%M%SZ).log\"",
        "echo \"[$(date -Is)] waiting for heimdall-overnight-broadcast-ablation\" | tee -a \"$chain_log\"",
        "while tmux has-session -t heimdall-overnight-broadcast-ablation 2>/dev/null; do",
        "  sleep 30",
        "done",
        "echo \"[$(date -Is)] LLM matrix finished; starting deterministic baselines\" | tee -a \"$chain_log\"",
        "bash ai-society/runs/overnight-broadcast-ablation-deterministic/run_overnight_deterministic_baselines.sh | tee -a \"$chain_log\"",
    ]
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)


if __name__ == "__main__":
    main()
