from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/large-bid-generalization-matrix")
RUN_ROOT = Path("ai-society/runs/large-bid-generalization-matrix")
PRIORITY_LOG_ROOT = Path("ai-society/runs/priority-calibration-matrix/logs")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"
MODEL = "Qwen/Qwen3-32B"

WINDOWS_24 = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr03-1430": "2026-04-03T14:30:00Z",
    "apr05-1030": "2026-04-05T10:30:00Z",
    "apr06-1300": "2026-04-06T13:00:00Z",
    "apr07-1715": "2026-04-07T17:15:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
    "apr22-0830": "2026-04-22T08:30:00Z",
    "apr27-1730": "2026-04-27T17:30:00Z",
}

FULL_DAYS_96 = {
    "apr02": "2026-04-02T00:00:00Z",
    "apr03": "2026-04-03T00:00:00Z",
    "apr07": "2026-04-07T00:00:00Z",
    "apr09": "2026-04-09T00:00:00Z",
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

CORE_S12_RUNS = [
    ("s12-balanced", window_slug, sizing_mode, "core_s12")
    for window_slug in ("apr02-0530", "apr09-1830", "apr13-0015")
    for sizing_mode in ("medium", "large")
]

HISTORICAL_CORE_RUNS = [
    (society_slug, window_slug, "large", "historical_core")
    for window_slug in ("apr03-1430", "apr05-1030")
    for society_slug in ("s06-actioncore", "s12-balanced", "s20-mixed")
]

BREADTH_RUNS = [
    (society_slug, window_slug, "large", "breadth_sparse")
    for window_slug in ("apr06-1300", "apr07-1715", "apr22-0830", "apr27-1730")
    for society_slug in ("s06-actioncore", "s20-mixed")
]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    configs: list[Path] = []
    seen: set[str] = set()

    for society_slug, window_slug, sizing_mode, purpose in [
        *CORE_S12_RUNS,
        *HISTORICAL_CORE_RUNS,
        *BREADTH_RUNS,
    ]:
        run_id = f"lbg-{society_slug}-{sizing_mode}-{window_slug}-seed42-q32"
        path = ROOT / "ticks24" / purpose / society_slug / sizing_mode / f"{run_id}.yaml"
        _write_config(path, _config_24(run_id, society_slug, window_slug, sizing_mode), seen)
        configs.append(path)

    for day_slug in FULL_DAYS_96:
        run_id = f"lbg-mixed20-large-{day_slug}-96-real-controls-q32"
        path = ROOT / "ticks96" / "s20-mixed" / "large" / f"{run_id}.yaml"
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


def _sizing_fields(sizing_mode: str) -> dict[str, Any]:
    if sizing_mode == "medium":
        return {
            "candidate_sizing_mode": "medium",
            "candidate_sizing_cap_fraction": 0.5,
            "candidate_sizing_min_mwh": 0.25,
            "candidate_sizing_max_candidates": 8,
        }
    if sizing_mode == "large":
        return {
            "candidate_sizing_mode": "large",
            "candidate_sizing_cap_fraction": 1.0,
            "candidate_sizing_min_mwh": 0.25,
            "candidate_sizing_max_candidates": 8,
        }
    raise ValueError(f"unknown sizing mode: {sizing_mode}")


def _base_payload(run_id: str, ticks: int, start_timestamp: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": 42,
        "forecaster_seed": 42,
        "zone": "DK1",
        "ticks": ticks,
        "start_timestamp": start_timestamp,
        "forecaster_backend": "f8",
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "objective": "bid_seeking",
        "ablation_strategy": "comm_broadcast_digest",
        "scenario_id": "p2h_dk1_pypsa",
        "max_tool_rounds": 6,
        "data_start": "2026-04-01T00:00:00Z",
        "data_end": "2026-05-01T00:00:00Z",
        "context_dataset_dir": CONTEXT_DIR,
        "data_cache_dir": f"{CONTEXT_DIR}/source_cache",
        "default_lookback_hours": 24,
        "cache_refresh": False,
        "output_dir": str(RUN_ROOT),
        "reviewer_mode": "code_only",
    }


def _config_24(run_id: str, society_slug: str, window_slug: str, sizing_mode: str) -> dict[str, Any]:
    society = SOCIETIES[society_slug]
    payload = {
        **_base_payload(run_id, 24, WINDOWS_24[window_slug]),
        "agent_count": society["agent_count"],
        "persona_profile": society["persona_profile"],
        "tool_policy": "p2h_only_simulator",
        **_sizing_fields(sizing_mode),
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


def _config_96(run_id: str, day_slug: str) -> dict[str, Any]:
    return {
        **_base_payload(run_id, 96, FULL_DAYS_96[day_slug]),
        "agent_count": 20,
        "persona_profile": "mixed_expert_20_sideaware",
        "tool_policy": "asset_simulator_v1",
        "asset_simulator_mode": "dual_compare_real_controls",
        "asset_proxy_style": "market",
        "simulator_max_concurrency": 8,
        "memory_enabled": True,
        "memory_bank_path": MEMORY_BANK,
        "memory_max_items_per_agent": 5,
        "memory_max_prompt_chars": 2400,
        **_sizing_fields("large"),
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
        candidates = [RUN_ROOT / run_id, Path("evaluations") / run_id]
        collisions.extend(str(path) for path in candidates if path.exists())
    if collisions:
        raise RuntimeError("refusing to generate duplicate large-bid generalization outputs:\n" + "\n".join(collisions))


def _write_config_list(configs: list[Path]) -> None:
    (ROOT / "config-list.txt").write_text("".join(f"{path}\n" for path in configs), encoding="utf-8")


def _write_manifest(configs: list[Path]) -> None:
    payload = {
        "run_count": len(configs),
        "seed": 42,
        "model": MODEL,
        "forecaster_backend": "f8",
        "ablation_strategy": "comm_broadcast_digest",
        "sizing_arms": {
            "medium": "candidate quantities around 25-50% of archetype cap",
            "large": "candidate quantities up to archetype cap",
        },
        "config_list": str(ROOT / "config-list.txt"),
        "run_root": str(RUN_ROOT),
        "runs": [
            {
                "run_id": path.stem,
                "config": str(path),
                "ticks": 96 if "-96-" in path.stem else 24,
                "purpose": path.parts[len(ROOT.parts) + 1],
                "candidate_sizing_mode": "medium" if "-medium-" in path.stem else "large",
            }
            for path in configs
        ],
        "chain_after": {
            "session": "priority-calibration",
            "summary_root": str(PRIORITY_LOG_ROOT),
            "required_completed": 6,
            "required_failed": 0,
        },
        "known_existing_large_bid_cells_not_duplicated": [
            "fco-size s06/s20 medium+large apr02/apr09/apr13 seed42 q32",
        ],
    }
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_runners() -> None:
    runner = RUN_ROOT / "run_large_bid_generalization_matrix.sh"
    chain = RUN_ROOT / "chain_after_priority_calibration.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd /home/ucloud/heimdall",
                "export PYTHONPATH='.'",
                "mkdir -p ai-society/runs/large-bid-generalization-matrix/logs",
                "log_dir=\"ai-society/runs/large-bid-generalization-matrix/logs/$(date -u +%Y%m%dT%H%M%SZ)\"",
                "mkdir -p \"$log_dir\"",
                "echo \"[$(date -Is)] large-bid generalization matrix start log_dir=$log_dir\"",
                "uv run python ai-society/run_long_model_society_matrix.py \\",
                "  --config-list ai-society/configs/large-bid-generalization-matrix/config-list.txt \\",
                "  --log-dir \"$log_dir\" \\",
                "  --continue-on-failure \\",
                "  > \"$log_dir/controller.stdout.log\" 2>&1",
                "echo \"[$(date -Is)] large-bid generalization matrix complete log_dir=$log_dir\"",
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
                "priority_pattern='run_long_model_society_matrix.py --config-list ai-society/configs/priority-calibration-matrix/config-list.txt'",
                "while pgrep -af \"$priority_pattern\" >/dev/null; do",
                "  echo \"[$(date -Is)] waiting for priority-calibration matrix to finish\"",
                "  sleep 300",
                "done",
                "latest_summary=$(find ai-society/runs/priority-calibration-matrix/logs -maxdepth 2 -name summary.json -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)",
                "if [[ -z \"${latest_summary:-}\" ]]; then",
                "  echo \"[$(date -Is)] skip: no priority-calibration summary found\"",
                "  exit 1",
                "fi",
                "completed=$(uv run python -c \"import json,sys; print(json.load(open(sys.argv[1])).get('completed'))\" \"$latest_summary\")",
                "failed=$(uv run python -c \"import json,sys; print(json.load(open(sys.argv[1])).get('failed'))\" \"$latest_summary\")",
                "if [[ \"$completed\" != \"6\" || \"$failed\" != \"0\" ]]; then",
                "  echo \"[$(date -Is)] skip: priority-calibration incomplete or failed summary=$latest_summary completed=$completed failed=$failed\"",
                "  exit 1",
                "fi",
                "echo \"[$(date -Is)] priority-calibration clean; launching large-bid generalization\"",
                "exec ai-society/runs/large-bid-generalization-matrix/run_large_bid_generalization_matrix.sh",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    chain.chmod(0o755)


if __name__ == "__main__":
    main()
