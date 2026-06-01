from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/single-agent-archetypes")
RUN_ROOT = Path("ai-society/runs/single-agent-archetypes")
CONTEXT_DIR = "data/cache/real_context/april_2026"

ARCHETYPES = ["p2h", "ev", "wind", "generator", "renewables", "retailer", "arbitrageur"]
WINDOWS = [
    ("apr02-0530", "2026-04-02T05:30:00Z", 24),
    ("apr09-1830", "2026-04-09T18:30:00Z", 24),
    ("apr13-0015", "2026-04-13T00:15:00Z", 24),
]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    configs: list[Path] = []
    for window_slug, start, ticks in WINDOWS:
        for archetype in ARCHETYPES:
            run_id = f"sat-1-{archetype}-{window_slug}-f8-q32"
            path = ROOT / f"{run_id}.yaml"
            path.write_text(yaml.safe_dump(_config(run_id, start, ticks, archetype), sort_keys=False), encoding="utf-8")
            configs.append(path)
    gpu0 = [path for idx, path in enumerate(configs) if idx % 2 == 0]
    gpu1 = [path for idx, path in enumerate(configs) if idx % 2 == 1]
    (ROOT / "gpu0.txt").write_text("".join(str(path) + "\n" for path in gpu0), encoding="utf-8")
    (ROOT / "gpu1.txt").write_text("".join(str(path) + "\n" for path in gpu1), encoding="utf-8")
    manifest = {"run_count": len(configs), "windows": WINDOWS, "archetypes": ARCHETYPES}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_runner()
    print(json.dumps({"ok": True, "run_count": len(configs), "manifest": str(ROOT / "manifest.json")}, indent=2))


def _config(run_id: str, start: str, ticks: int, archetype: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": 42,
        "zone": "DK1",
        "agent_count": 1,
        "archetype_cycle": [archetype],
        "ticks": ticks,
        "start_timestamp": start,
        "forecaster_backend": "f8",
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "objective": "bid_seeking",
        "ablation_strategy": "diverse_action_society",
        "persona_profile": "default",
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
        "llm": {
            "enabled": True,
            "model": "Qwen/Qwen3-32B",
            "temperature": 0.2,
            "max_tokens": 1000,
            "timeout_seconds": 180,
            "max_concurrency": 1,
        },
    }


def _write_runner() -> None:
    script = RUN_ROOT / "run_single_agent_archetypes.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /home/ucloud/heimdall",
        "export PYTHONPATH='.'",
        "mkdir -p ai-society/runs/single-agent-archetypes/logs",
        "log_dir=\"ai-society/runs/single-agent-archetypes/logs/$(date -u +%Y%m%dT%H%M%SZ)\"",
        "mkdir -p \"$log_dir\"",
        "python ai-society/run_market_intelligence_stage.py --stage single-agent-archetypes --gpu gpu0 --base-url http://127.0.0.1:8000/v1 --config-list ai-society/configs/single-agent-archetypes/gpu0.txt --log-dir \"$log_dir\" > \"$log_dir/gpu0.stdout.log\" 2>&1 &",
        "pid0=$!",
        "python ai-society/run_market_intelligence_stage.py --stage single-agent-archetypes --gpu gpu1 --base-url http://127.0.0.1:8001/v1 --config-list ai-society/configs/single-agent-archetypes/gpu1.txt --log-dir \"$log_dir\" > \"$log_dir/gpu1.stdout.log\" 2>&1 &",
        "pid1=$!",
        "echo \"$pid0 $pid1\" > \"$log_dir/pids.txt\"",
        "echo \"launched single-agent archetypes log_dir=$log_dir gpu0=$pid0 gpu1=$pid1\"",
        "wait",
    ]
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)


if __name__ == "__main__":
    main()
