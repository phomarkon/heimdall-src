from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/follow-on-breadth-matrix")
RUN_ROOT = Path("ai-society/runs/follow-on-breadth-matrix")
FOA_LOG_ROOT = Path("ai-society/runs/final-option-a-model-scale/logs")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"
MODEL = "Qwen/Qwen3-32B"

WINDOWS_24 = {
    "apr06-1300": "2026-04-06T13:00:00Z",
    "apr22-0830": "2026-04-22T08:30:00Z",
    "apr27-1730": "2026-04-27T17:30:00Z",
}

FULL_DAYS_96 = {
    "apr06": "2026-04-06T00:00:00Z",
    "apr07": "2026-04-07T00:00:00Z",
}

SOCIETIES_24 = {
    "s06-actioncore": {
        "agent_count": 6,
        "persona_profile": "action_core_8",
        "max_concurrency": 8,
        "per_endpoint_max_concurrency": 4,
        "max_tokens": 1000,
    },
    "s20-mixed": {
        "agent_count": 20,
        "persona_profile": "mixed_expert_20_sideaware",
        "max_concurrency": 16,
        "per_endpoint_max_concurrency": 8,
        "max_tokens": 1000,
    },
}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    configs: list[Path] = []
    seen: set[str] = set()

    for window_slug in WINDOWS_24:
        for society_slug in ("s06-actioncore", "s20-mixed"):
            run_id = f"fob-{society_slug}-bcast-{window_slug}-seed42-q32"
            path = ROOT / "ticks24" / society_slug / f"{run_id}.yaml"
            _write_config(path, _config_24(run_id, society_slug, window_slug), seen)
            configs.append(path)

    for day_slug in FULL_DAYS_96:
        run_id = f"fob-mixed20-{day_slug}-96-real-controls-q32"
        path = ROOT / "ticks96" / "mixed20-real-controls" / f"{run_id}.yaml"
        _write_config(path, _config_96(run_id, day_slug), seen)
        configs.append(path)

    _assert_no_existing_outputs(seen)
    _write_config_list(configs)
    _write_manifest(configs)
    _write_runners()

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


def _config_24(run_id: str, society_slug: str, window_slug: str) -> dict[str, Any]:
    society = SOCIETIES_24[society_slug]
    return {
        "run_id": run_id,
        "seed": 42,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": society["agent_count"],
        "ticks": 24,
        "start_timestamp": WINDOWS_24[window_slug],
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


def _config_96(run_id: str, day_slug: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": 42,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": 20,
        "ticks": 96,
        "start_timestamp": FULL_DAYS_96[day_slug],
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
        candidates = [
            RUN_ROOT / run_id,
            Path("evaluations") / run_id,
        ]
        collisions.extend(str(path) for path in candidates if path.exists())
    if collisions:
        raise RuntimeError("refusing to generate duplicate follow-on outputs:\n" + "\n".join(collisions))


def _write_config_list(configs: list[Path]) -> None:
    (ROOT / "config-list.txt").write_text("".join(f"{path}\n" for path in configs), encoding="utf-8")


def _write_manifest(configs: list[Path]) -> None:
    payload = {
        "run_count": len(configs),
        "seed": 42,
        "model": MODEL,
        "forecaster_backend": "f8",
        "ablation_strategy": "comm_broadcast_digest",
        "run_root": str(RUN_ROOT),
        "config_list": str(ROOT / "config-list.txt"),
        "runs": [
            {
                "run_id": path.stem,
                "config": str(path),
                "ticks": 96 if "-96-" in path.stem else 24,
                "priority": 2 if "-96-" in path.stem else 1,
            }
            for path in configs
        ],
        "chain_after": {
            "session": "foa-model-scale",
            "summary_root": str(FOA_LOG_ROOT),
            "required_completed": 12,
            "required_failed": 0,
        },
    }
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_runners() -> None:
    runner = RUN_ROOT / "run_follow_on_breadth_matrix.sh"
    chain = RUN_ROOT / "chain_after_final_option_a.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd /home/ucloud/heimdall",
                "export PYTHONPATH='.'",
                "mkdir -p ai-society/runs/follow-on-breadth-matrix/logs",
                "log_dir=\"ai-society/runs/follow-on-breadth-matrix/logs/$(date -u +%Y%m%dT%H%M%SZ)\"",
                "mkdir -p \"$log_dir\"",
                "echo \"[$(date -Is)] follow-on breadth matrix start log_dir=$log_dir\"",
                "uv run python ai-society/run_long_model_society_matrix.py \\",
                "  --config-list ai-society/configs/follow-on-breadth-matrix/config-list.txt \\",
                "  --log-dir \"$log_dir\" \\",
                "  --continue-on-failure \\",
                "  > \"$log_dir/controller.stdout.log\" 2>&1",
                "echo \"[$(date -Is)] follow-on breadth matrix complete log_dir=$log_dir\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)

    chain.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd /home/ucloud/heimdall",
                "export PYTHONPATH='.'",
                "chain_log=\"ai-society/runs/follow-on-breadth-matrix/chain-after-final-option-a.log\"",
                "mkdir -p \"$(dirname \"$chain_log\")\"",
                "echo \"[$(date -Is)] waiting for foa-model-scale to finish\" | tee -a \"$chain_log\"",
                "while tmux has-session -t foa-model-scale 2>/dev/null; do",
                "  sleep 60",
                "done",
                "echo \"[$(date -Is)] foa-model-scale exited; checking latest resume summary\" | tee -a \"$chain_log\"",
                "uv run python - <<'PY' 2>&1 | tee -a \"$chain_log\"",
                "import json",
                "from pathlib import Path",
                "",
                "root = Path('ai-society/runs/final-option-a-model-scale/logs')",
                "summaries = sorted(root.glob('*resume-run4/summary.json'), key=lambda p: p.stat().st_mtime)",
                "if not summaries:",
                "    raise SystemExit('no final option A resume summary found')",
                "summary_path = summaries[-1]",
                "summary = json.loads(summary_path.read_text(encoding='utf-8'))",
                "print(json.dumps({'summary': str(summary_path), **summary}, indent=2, sort_keys=True))",
                "if summary.get('completed') != 12 or summary.get('failed') != 0:",
                "    raise SystemExit('final option A did not finish cleanly; follow-on not started')",
                "PY",
                "echo \"[$(date -Is)] final option A clean; starting follow-on-breadth tmux\" | tee -a \"$chain_log\"",
                "tmux new-session -d -s follow-on-breadth 'cd /home/ucloud/heimdall && bash ai-society/runs/follow-on-breadth-matrix/run_follow_on_breadth_matrix.sh'",
                "echo \"[$(date -Is)] started tmux session follow-on-breadth\" | tee -a \"$chain_log\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    chain.chmod(0o755)


if __name__ == "__main__":
    main()
