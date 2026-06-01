from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path("ai-society/configs/deliberation-protocol-20260519")
RUN_ROOT = Path("ai-society/runs/deliberation-protocol-20260519")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MODEL = "Qwen/Qwen3-32B"

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}

SOCIETIES = {
    "s06-actioncore": {
        "agent_count": 6,
        "persona_profile": "action_core_8",
        "max_concurrency": 6,
        "per_endpoint_max_concurrency": 3,
        "max_tokens": 1000,
    },
    "s12-balanced": {
        "agent_count": 12,
        "persona_profile": "balanced_intelligence",
        "max_concurrency": 12,
        "per_endpoint_max_concurrency": 6,
        "max_tokens": 1000,
    },
}

RAMP = [
    ("stage1-s06-2", "s06-actioncore", "apr02-0530", 2),
    ("stage2-s06-5", "s06-actioncore", "apr02-0530", 5),
    ("stage3-s06-10", "s06-actioncore", "apr02-0530", 10),
    ("stage4-s12-5", "s12-balanced", "apr02-0530", 5),
    ("stage5-s12-10", "s12-balanced", "apr02-0530", 10),
]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    configs: list[Path] = []
    ramp_configs: list[Path] = []
    final_delib_configs: list[Path] = []
    final_comparator_configs: list[Path] = []

    for stage_slug, society_slug, window_slug, ticks in RAMP:
        run_id = f"dproto-{stage_slug}-{window_slug}-seed42-q32"
        path = ROOT / "ramp" / f"{run_id}.yaml"
        _write_config(path, _config(run_id, society_slug, window_slug, ticks, arm="delib-context"))
        configs.append(path)
        ramp_configs.append(path)

    for window_slug in WINDOWS:
        for society_slug in SOCIETIES:
            delib_id = f"dproto-final-{society_slug}-delib-{window_slug}-24-seed42-q32"
            delib_path = ROOT / "final" / society_slug / f"{delib_id}.yaml"
            _write_config(delib_path, _config(delib_id, society_slug, window_slug, 24, arm="delib-context"))
            configs.append(delib_path)
            final_delib_configs.append(delib_path)

            comparator_id = f"dproto-final-{society_slug}-guardedfull-{window_slug}-24-seed42-q32"
            if not (RUN_ROOT / comparator_id).exists():
                comparator_path = ROOT / "final" / society_slug / f"{comparator_id}.yaml"
                _write_config(comparator_path, _config(comparator_id, society_slug, window_slug, 24, arm="guarded-full"))
                configs.append(comparator_path)
                final_comparator_configs.append(comparator_path)

    _write_list(ROOT / "ramp.txt", ramp_configs)
    _write_list(ROOT / "ramp-s06.txt", [path for path in ramp_configs if "s06" in path.name])
    _write_list(ROOT / "ramp-s12.txt", [path for path in ramp_configs if "s12" in path.name])
    _write_list(ROOT / "final-delib.txt", final_delib_configs)
    _write_list(ROOT / "final-comparators.txt", final_comparator_configs)
    _write_list(ROOT / "all.txt", configs)
    _write_manifest(configs, ramp_configs, final_delib_configs, final_comparator_configs)
    _write_runner()
    print(json.dumps({"ok": True, "run_count": len(configs), "root": str(ROOT)}, indent=2))


def _config(run_id: str, society_slug: str, window_slug: str, ticks: int, *, arm: str) -> dict[str, Any]:
    society = SOCIETIES[society_slug]
    delib = arm == "delib-context"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "seed": 42,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": society["agent_count"],
        "ticks": ticks,
        "start_timestamp": WINDOWS[window_slug],
        "forecaster_backend": "f8",
        "forecaster_routing_mode": "run_level",
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "objective": "bid_seeking",
        "ablation_strategy": "comm_deliberation_protocol" if delib else "comm_broadcast_digest_priority_calibration",
        "persona_profile": society["persona_profile"],
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "asset_simulator_v1",
        "asset_simulator_mode": "dual_compare_real_controls",
        "candidate_sizing_mode": "medium",
        "candidate_sizing_max_candidates": 8,
        "preprobe_mode": "none" if delib else "full",
        "final_bid_guard": "simulator_exact_match",
        "max_tool_rounds": 6,
        "simulator_max_concurrency": 8,
        "deliberation_inquiry_rounds": 1,
        "deliberation_action_rounds": 1,
        "deliberation_min_tool_calls": 1,
        "deliberation_require_action_probe": True,
        "deliberation_max_peer_notes": 12,
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
    return payload


def _write_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_list(path: Path, configs: list[Path]) -> None:
    path.write_text("".join(f"{config}\n" for config in configs), encoding="utf-8")


def _write_manifest(
    configs: list[Path],
    ramp_configs: list[Path],
    final_delib_configs: list[Path],
    final_comparator_configs: list[Path],
) -> None:
    payload = {
        "run_count": len(configs),
        "model": MODEL,
        "run_root": str(RUN_ROOT),
        "fast_validation_ladder": RAMP,
        "acceptance_gates": {
            "inquiry_tool_call_rate_min": 0.8,
            "deliberation_note_rate_min": 0.7,
            "action_probe_compliance_rate_min": 0.5,
        },
        "config_lists": {
            "ramp": str(ROOT / "ramp.txt"),
            "ramp_s06": str(ROOT / "ramp-s06.txt"),
            "ramp_s12": str(ROOT / "ramp-s12.txt"),
            "final_delib": str(ROOT / "final-delib.txt"),
            "final_comparators": str(ROOT / "final-comparators.txt"),
            "all": str(ROOT / "all.txt"),
        },
        "runs": [str(config) for config in configs],
        "ramp_runs": [str(config) for config in ramp_configs],
        "final_delib_runs": [str(config) for config in final_delib_configs],
        "final_comparator_runs": [str(config) for config in final_comparator_configs],
    }
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_runner() -> None:
    script = RUN_ROOT / "run_deliberation_ramp_then_matrix.sh"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd /home/ucloud/heimdall",
                "export PYTHONPATH='.'",
                "mkdir -p ai-society/runs/deliberation-protocol-20260519/logs",
                "log_dir=\"ai-society/runs/deliberation-protocol-20260519/logs/$(date -u +%Y%m%dT%H%M%SZ)\"",
                "mkdir -p \"$log_dir\"",
                "echo \"[$(date -Is)] deliberation S06 ramp start\"",
                "uv run python ai-society/run_long_model_society_matrix.py --config-list ai-society/configs/deliberation-protocol-20260519/ramp-s06.txt --log-dir \"$log_dir\" > \"$log_dir/ramp-s06.stdout.log\" 2>&1",
                "uv run python tools/experiments/summarize_deliberation_protocol.py ai-society/runs/deliberation-protocol-20260519/dproto-stage*-s06-*",
                "echo \"[$(date -Is)] deliberation S12 ramp start\"",
                "uv run python ai-society/run_long_model_society_matrix.py --config-list ai-society/configs/deliberation-protocol-20260519/ramp-s12.txt --log-dir \"$log_dir\" > \"$log_dir/ramp-s12.stdout.log\" 2>&1",
                "uv run python tools/experiments/summarize_deliberation_protocol.py ai-society/runs/deliberation-protocol-20260519/dproto-stage*-s12-*",
                "echo \"[$(date -Is)] deliberation final matrix start\"",
                "uv run python ai-society/run_long_model_society_matrix.py --config-list ai-society/configs/deliberation-protocol-20260519/final-delib.txt --log-dir \"$log_dir\" --continue-on-failure > \"$log_dir/final-delib.stdout.log\" 2>&1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


if __name__ == "__main__":
    main()
