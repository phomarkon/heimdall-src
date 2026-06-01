from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/market-intelligence-full-suite")
RUNS_ROOT = Path("ai-society/runs/market-intelligence-full-suite")


@dataclass(frozen=True)
class Window:
    slug: str
    start: str
    ticks: int
    family: str
    reason: str


WINDOWS = [
    Window("apr02-1200", "2026-04-02T12:00:00Z", 24, "known", "strong profit and good side precision"),
    Window("apr02-0530", "2026-04-02T05:30:00Z", 48, "known", "strong profit and many actions"),
    Window("apr03-1430", "2026-04-03T14:30:00Z", 48, "known", "new profitable screened run"),
    Window("apr09-1830", "2026-04-09T18:30:00Z", 48, "known", "best absolute profit"),
    Window("apr13-0015", "2026-04-13T00:15:00Z", 48, "known", "watch-heavy oracle window"),
    Window("roll1-apr27-1730", "2026-04-27T17:30:00Z", 48, "rolling", "pre-registered agent-context score=0.929688"),
    Window("roll2-apr04-0600", "2026-04-04T06:00:00Z", 48, "rolling", "pre-registered agent-context score=0.925000"),
    Window("roll3-apr07-2230", "2026-04-07T22:30:00Z", 48, "rolling", "pre-registered agent-context score=0.920312"),
]

SOCIETIES = [
    {"size": 5, "slug": "diverse5", "profile": "diverse_action", "agent_count": 5, "max_concurrency": 5},
    {"size": 8, "slug": "expert8", "profile": "diverse_expert_action", "agent_count": 8, "max_concurrency": 6},
    {"size": 12, "slug": "balanced12", "profile": "balanced_intelligence", "agent_count": 12, "max_concurrency": 8},
]

COMMUNICATION = [
    ("independent", "diverse_action_society"),
    ("broadcast", "comm_broadcast_digest"),
    ("peer", "comm_peer_signal"),
    ("retry", "comm_retry_council"),
]

FORECASTERS = ["f8", "f3_ensemble", "f0"]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (RUNS_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {forecaster: [] for forecaster in FORECASTERS}
    deterministic: list[str] = []
    seen: set[str] = set()

    for forecaster in FORECASTERS:
        stage_dir = ROOT / f"stage-{forecaster}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        for window in WINDOWS:
            for society in SOCIETIES:
                for comm_slug, strategy in COMMUNICATION:
                    run_id = f"mi-{society['size']}-{society['slug']}-{comm_slug}-{window.slug}-{forecaster}-q32"
                    path = stage_dir / f"{run_id}.yaml"
                    _write_yaml(path, _llm_config(run_id, forecaster, window, society, strategy))
                    _record(path, run_id, seen)
                    written[forecaster].append(str(path))

    baseline_dir = ROOT / "deterministic-baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for forecaster in FORECASTERS:
        for window in WINDOWS:
            run_id = f"mi-detbest-diverse5-{window.slug}-{forecaster}"
            path = baseline_dir / f"{run_id}.yaml"
            _write_yaml(path, _deterministic_config(run_id, forecaster, window))
            _record(path, run_id, seen)
            deterministic.append(str(path))

    split_files: dict[str, dict[str, str]] = {}
    for forecaster, configs in written.items():
        split_files[forecaster] = _write_splits(forecaster, configs)
    det_splits = _write_splits("deterministic", deterministic)

    manifest = {
        "schema_version": "1.0.0",
        "suite": "market-intelligence-full-suite",
        "run_count": sum(len(v) for v in written.values()),
        "deterministic_baseline_count": len(deterministic),
        "stages": written,
        "splits": split_files,
        "deterministic_splits": det_splits,
        "windows": [window.__dict__ for window in WINDOWS],
        "societies": SOCIETIES,
        "communication": [{"slug": slug, "ablation_strategy": strategy} for slug, strategy in COMMUNICATION],
        "forecasters": FORECASTERS,
        "context_dataset_dir": "data/cache/real_context/april_2026",
        "truth_dir": "data/cache/evaluation_truth/april_2026",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_run_scripts(split_files, det_splits)
    print(json.dumps({"ok": True, "manifest": str(ROOT / "manifest.json"), "run_count": manifest["run_count"], "deterministic_baseline_count": len(deterministic)}, indent=2))


def _base_config(run_id: str, forecaster: str, window: Window, *, chooser_mode: str, agent_count: int, profile: str, max_concurrency: int, ablation_strategy: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": 42,
        "zone": "DK1",
        "agent_count": agent_count,
        "ticks": window.ticks,
        "start_timestamp": window.start,
        "forecaster_backend": forecaster,
        "chooser_mode": chooser_mode,
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "objective": "bid_seeking",
        "ablation_strategy": ablation_strategy,
        "persona_profile": profile,
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "p2h_only_simulator",
        "max_tool_rounds": 6,
        "data_start": "2026-04-01T00:00:00Z",
        "data_end": "2026-05-01T00:00:00Z",
        "context_dataset_dir": "data/cache/real_context/april_2026",
        "data_cache_dir": "data/cache/real_context/april_2026/source_cache",
        "default_lookback_hours": 24,
        "cache_refresh": False,
        "output_dir": "ai-society/runs",
        "llm": {
            "enabled": True,
            "model": "Qwen/Qwen3-32B",
            "temperature": 0.2,
            "max_tokens": 1000,
            "timeout_seconds": 180,
            "max_concurrency": max_concurrency,
        },
    }


def _llm_config(run_id: str, forecaster: str, window: Window, society: dict[str, Any], strategy: str) -> dict[str, Any]:
    return _base_config(
        run_id,
        forecaster,
        window,
        chooser_mode="llm",
        agent_count=int(society["agent_count"]),
        profile=str(society["profile"]),
        max_concurrency=int(society["max_concurrency"]),
        ablation_strategy=strategy,
    )


def _deterministic_config(run_id: str, forecaster: str, window: Window) -> dict[str, Any]:
    config = _base_config(
        run_id,
        forecaster,
        window,
        chooser_mode="deterministic_best_accepted",
        agent_count=5,
        profile="diverse_action",
        max_concurrency=1,
        ablation_strategy="diverse_action_society",
    )
    config["llm"]["enabled"] = False
    return config


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _record(path: Path, run_id: str, seen: set[str]) -> None:
    if run_id in seen:
        raise RuntimeError(f"duplicate run_id: {run_id}")
    seen.add(run_id)


def _write_splits(stage: str, configs: list[str]) -> dict[str, str]:
    out = {}
    for gpu, selected in {
        "gpu0": configs[0::2],
        "gpu1": configs[1::2],
    }.items():
        path = ROOT / f"{stage}-{gpu}.txt"
        path.write_text("\n".join(selected) + "\n", encoding="utf-8")
        out[gpu] = str(path)
    return out


def _write_run_scripts(split_files: dict[str, dict[str, str]], det_splits: dict[str, str]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'stage="${1:-f8}"',
        'case "$stage" in',
    ]
    for stage, splits in split_files.items():
        lines.extend(
            [
                f"  {stage})",
                f"    gpu0_list='{splits['gpu0']}'",
                f"    gpu1_list='{splits['gpu1']}'",
                "    ;;",
            ]
        )
    lines.extend(
        [
            "  deterministic)",
            f"    gpu0_list='{det_splits['gpu0']}'",
            f"    gpu1_list='{det_splits['gpu1']}'",
            "    ;;",
            "  *) echo \"unknown stage: $stage\" >&2; exit 2 ;;",
            "esac",
            "mkdir -p ai-society/runs/market-intelligence-full-suite/logs",
            "log_dir=\"ai-society/runs/market-intelligence-full-suite/logs/$stage-$(date -u +%Y%m%dT%H%M%SZ)\"",
            "mkdir -p \"$log_dir\"",
            "python ai-society/run_market_intelligence_stage.py --stage \"$stage\" --gpu gpu0 --base-url http://127.0.0.1:8000/v1 --config-list \"$gpu0_list\" --log-dir \"$log_dir\" > \"$log_dir/gpu0.stdout.log\" 2>&1 &",
            "pid0=$!",
            "python ai-society/run_market_intelligence_stage.py --stage \"$stage\" --gpu gpu1 --base-url http://127.0.0.1:8001/v1 --config-list \"$gpu1_list\" --log-dir \"$log_dir\" > \"$log_dir/gpu1.stdout.log\" 2>&1 &",
            "pid1=$!",
            "echo \"$pid0 $pid1\" > \"$log_dir/pids.txt\"",
            "echo \"launched $stage: gpu0=$pid0 gpu1=$pid1 log_dir=$log_dir\"",
        ]
    )
    script = RUNS_ROOT / "run_stage.sh"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)


if __name__ == "__main__":
    main()
