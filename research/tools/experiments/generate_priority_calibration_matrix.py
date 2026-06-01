from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path("ai-society/configs/priority-calibration-matrix")
RUN_ROOT = Path("ai-society/runs/priority-calibration-matrix")
FOLLOW_LOG_ROOT = Path("ai-society/runs/follow-on-breadth-matrix/logs")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"
MODEL = "Qwen/Qwen3-32B"

WINDOWS = {
    "apr02": "2026-04-02T00:00:00Z",
    "apr03": "2026-04-03T00:00:00Z",
    "apr07": "2026-04-07T00:00:00Z",
    "apr09": "2026-04-09T00:00:00Z",
}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    configs: list[Path] = []
    seen: set[str] = set()
    for day_slug in WINDOWS:
        run_id = f"pcm-mixed20-{day_slug}-96-priority-real-q32"
        path = ROOT / "minimum" / f"{run_id}.yaml"
        _write_config(path, _config(run_id, day_slug, agent_count=20, persona_profile="mixed_expert_20_sideaware"), seen)
        configs.append(path)

    stretch = [
        (
            "pcm-mixed20-apr02-96-priority-proxy-q32",
            "apr02",
            20,
            "mixed_expert_20_sideaware",
            "dual_compare_proxy_controls",
        ),
        (
            "pcm-mixed18-apr09-96-priority-real-q32",
            "apr09",
            18,
            "mixed_expert_18_sideaware",
            "dual_compare_real_controls",
        ),
    ]
    for run_id, day_slug, agent_count, persona_profile, asset_mode in stretch:
        path = ROOT / "stretch" / f"{run_id}.yaml"
        _write_config(path, _config(run_id, day_slug, agent_count=agent_count, persona_profile=persona_profile, asset_mode=asset_mode), seen)
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
                "minimum_run_count": 4,
                "stretch_run_count": 2,
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


def _config(
    run_id: str,
    day_slug: str,
    *,
    agent_count: int,
    persona_profile: str,
    asset_mode: str = "dual_compare_real_controls",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": 42,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": agent_count,
        "ticks": 96,
        "start_timestamp": WINDOWS[day_slug],
        "forecaster_backend": "f8",
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "objective": "bid_seeking",
        "ablation_strategy": "comm_broadcast_digest_priority_calibration",
        "persona_profile": persona_profile,
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "asset_simulator_v1",
        "asset_simulator_mode": asset_mode,
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
            "max_tokens": 640,
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
        raise RuntimeError("refusing to generate duplicate priority calibration outputs:\n" + "\n".join(collisions))


def _write_config_list(configs: list[Path]) -> None:
    (ROOT / "config-list.txt").write_text("".join(f"{path}\n" for path in configs), encoding="utf-8")


def _write_manifest(configs: list[Path]) -> None:
    payload = {
        "run_count": len(configs),
        "minimum_run_count": 4,
        "stretch_run_count": 2,
        "seed": 42,
        "model": MODEL,
        "forecaster_backend": "f8",
        "ablation_strategy": "comm_broadcast_digest_priority_calibration",
        "run_root": str(RUN_ROOT),
        "config_list": str(ROOT / "config-list.txt"),
        "success_criteria": {
            "top_10_capture_mean_min": 0.20,
            "top_12_capture_mean_min": 0.25,
            "top_24_capture_mean_min": 0.40,
            "lift_over_random_top_10_or_12_min": 1.50,
            "priority_field_coverage_min": 0.95,
            "verifier_realized_profit_breach_rate": 0.0,
        },
        "runs": [
            {
                "run_id": path.stem,
                "config": str(path),
                "ticks": 96,
                "tier": "minimum" if "/minimum/" in str(path) else "stretch",
            }
            for path in configs
        ],
        "chain_after": {
            "session": "follow-on-breadth",
            "summary_root": str(FOLLOW_LOG_ROOT),
            "required_completed": 8,
            "required_failed": 0,
        },
    }
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_runners() -> None:
    runner = RUN_ROOT / "run_priority_calibration_matrix.sh"
    chain = RUN_ROOT / "chain_after_follow_on_breadth.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd /home/ucloud/heimdall",
                "export PYTHONPATH='.'",
                "mkdir -p ai-society/runs/priority-calibration-matrix/logs",
                "log_dir=\"ai-society/runs/priority-calibration-matrix/logs/$(date -u +%Y%m%dT%H%M%SZ)\"",
                "mkdir -p \"$log_dir\"",
                "echo \"[$(date -Is)] priority calibration matrix start log_dir=$log_dir\"",
                "uv run python ai-society/run_long_model_society_matrix.py \\",
                "  --config-list ai-society/configs/priority-calibration-matrix/config-list.txt \\",
                "  --log-dir \"$log_dir\" \\",
                "  --continue-on-failure \\",
                "  > \"$log_dir/controller.stdout.log\" 2>&1",
                "echo \"[$(date -Is)] priority calibration matrix complete; evaluating top-k priority capture\"",
                "uv run python tools/evaluation/evaluate_priority_calibration.py \\",
                "  --config-list ai-society/configs/priority-calibration-matrix/config-list.txt \\",
                "  --output-dir evaluations/priority-calibration-matrix \\",
                "  > \"$log_dir/priority-eval.stdout.log\" 2>&1",
                "echo \"[$(date -Is)] priority calibration evaluation complete log_dir=$log_dir\"",
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
                "chain_log=\"ai-society/runs/priority-calibration-matrix/chain-after-follow-on-breadth.log\"",
                "mkdir -p \"$(dirname \"$chain_log\")\"",
                "echo \"[$(date -Is)] waiting for follow-on-breadth to finish\" | tee -a \"$chain_log\"",
                "while tmux has-session -t follow-on-breadth 2>/dev/null; do",
                "  sleep 60",
                "done",
                "echo \"[$(date -Is)] follow-on-breadth exited; checking latest summary\" | tee -a \"$chain_log\"",
                "uv run python - <<'PY' 2>&1 | tee -a \"$chain_log\"",
                "import json",
                "from pathlib import Path",
                "",
                "root = Path('ai-society/runs/follow-on-breadth-matrix/logs')",
                "summaries = sorted(root.glob('*/summary.json'), key=lambda p: p.stat().st_mtime)",
                "if not summaries:",
                "    raise SystemExit('no follow-on breadth summary found')",
                "summary_path = summaries[-1]",
                "summary = json.loads(summary_path.read_text(encoding='utf-8'))",
                "print(json.dumps({'summary': str(summary_path), **summary}, indent=2, sort_keys=True))",
                "if summary.get('completed') != 8 or summary.get('failed') != 0:",
                "    raise SystemExit('follow-on breadth did not finish cleanly; priority calibration not started')",
                "PY",
                "echo \"[$(date -Is)] follow-on breadth clean; starting priority-calibration tmux\" | tee -a \"$chain_log\"",
                "tmux new-session -d -s priority-calibration 'cd /home/ucloud/heimdall && bash ai-society/runs/priority-calibration-matrix/run_priority_calibration_matrix.sh'",
                "echo \"[$(date -Is)] started tmux session priority-calibration\" | tee -a \"$chain_log\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    chain.chmod(0o755)


if __name__ == "__main__":
    main()
