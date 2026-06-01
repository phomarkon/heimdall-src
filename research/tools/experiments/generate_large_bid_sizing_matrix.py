from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/final-core-overnight-matrix")
GROUP = ROOT / "large_bid_sizing"
RUN_ROOT = Path("ai-society/runs/final-core-overnight-matrix")
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

SIZING_ARMS = {
    "medium": 0.5,
    "large": 1.0,
}


def main() -> None:
    GROUP.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    configs: list[Path] = []
    seen: set[str] = set()
    for society_slug in SOCIETIES:
        for window_slug in WINDOWS:
            for sizing_mode, cap_fraction in SIZING_ARMS.items():
                run_id = f"fco-size-{society_slug}-{sizing_mode}-{window_slug}-seed42-q32"
                path = GROUP / society_slug / sizing_mode / f"{run_id}.yaml"
                _write_config(path, _config(run_id, society_slug, window_slug, sizing_mode, cap_fraction), seen)
                configs.append(path)

    _assert_no_existing_outputs(seen)
    config_list = ROOT / "large-bid-sizing-config-list.txt"
    config_list.write_text("".join(f"{path}\n" for path in configs), encoding="utf-8")
    _write_manifest(configs, config_list)
    _write_chain_script(config_list)
    print(json.dumps({"ok": True, "run_count": len(configs), "config_list": str(config_list)}, indent=2))


def _write_config(path: Path, payload: dict[str, Any], seen: set[str]) -> None:
    run_id = str(payload["run_id"])
    if run_id in seen:
        raise RuntimeError(f"duplicate run_id: {run_id}")
    seen.add(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _config(run_id: str, society_slug: str, window_slug: str, sizing_mode: str, cap_fraction: float) -> dict[str, Any]:
    society = SOCIETIES[society_slug]
    return {
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
        "persona_profile": society["persona_profile"],
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "p2h_only_simulator",
        "max_tool_rounds": 6,
        "candidate_sizing_mode": sizing_mode,
        "candidate_sizing_cap_fraction": cap_fraction,
        "candidate_sizing_min_mwh": 0.25,
        "candidate_sizing_max_candidates": 8,
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


def _assert_no_existing_outputs(run_ids: set[str]) -> None:
    collisions: list[str] = []
    for run_id in sorted(run_ids):
        candidates = [RUN_ROOT / run_id, Path("evaluations") / run_id]
        collisions.extend(str(path) for path in candidates if path.exists())
    if collisions:
        raise RuntimeError("refusing to generate duplicate large-bid sizing outputs:\n" + "\n".join(collisions))


def _write_manifest(configs: list[Path], config_list: Path) -> None:
    payload = {
        "run_count": len(configs),
        "model": MODEL,
        "forecaster_backend": "f8",
        "forecaster_seed": 42,
        "ablation_strategy": "comm_broadcast_digest",
        "purpose": "large_bid_sizing",
        "config_list": str(config_list),
        "run_root": str(RUN_ROOT),
        "target_wall_clock_hours": "6-8",
        "hard_wall_clock_cap_hours": 10,
        "sizing_arms": {
            "medium": "candidate quantities around 25-50% of archetype cap",
            "large": "candidate quantities up to archetype cap",
        },
        "runs": [
            {
                "run_id": path.stem,
                "config": str(path),
                "society": path.parts[-3],
                "candidate_sizing_mode": path.parts[-2],
            }
            for path in configs
        ],
    }
    (GROUP / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_chain_script(config_list: Path) -> None:
    script = RUN_ROOT / "chain_large_bid_sizing_after_final_core.sh"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd /home/ucloud/heimdall",
                "export PYTHONPATH='.'",
                "",
                "while pgrep -af \"run_long_model_society_matrix.py --config-list ai-society/configs/final-core-overnight-matrix/config-list.txt\" >/dev/null; do",
                "  echo \"[$(date -Is)] waiting for final-core-overnight-matrix to finish\"",
                "  sleep 300",
                "done",
                "",
                "log_dir=\"ai-society/runs/final-core-overnight-matrix/logs/large-bid-sizing-$(date -u +%Y%m%dT%H%M%SZ)\"",
                "mkdir -p \"$log_dir\"",
                "echo \"[$(date -Is)] large-bid sizing matrix start log_dir=$log_dir\"",
                "uv run python ai-society/run_long_model_society_matrix.py \\",
                f"  --config-list {config_list} \\",
                "  --log-dir \"$log_dir\" \\",
                "  --continue-on-failure \\",
                "  --skip-vllm-restart \\",
                "  > \"$log_dir/controller.stdout.log\" 2>&1 || {",
                "    echo \"[$(date -Is)] retrying without --skip-vllm-restart\"",
                "    uv run python ai-society/run_long_model_society_matrix.py \\",
                f"      --config-list {config_list} \\",
                "      --log-dir \"$log_dir-restart\" \\",
                "      --continue-on-failure \\",
                "      > \"$log_dir-restart/controller.stdout.log\" 2>&1",
                "  }",
                "echo \"[$(date -Is)] large-bid sizing matrix complete\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


if __name__ == "__main__":
    main()
