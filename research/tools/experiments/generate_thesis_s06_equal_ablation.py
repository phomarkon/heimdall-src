from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/thesis-s06-equal-ablation-20260519")
RUN_ROOT = Path("ai-society/runs/thesis-s06-equal-ablation-20260519")
CONTEXT_DIR = "data/cache/real_context/april_2026"
TRUTH_DIR = "data/cache/evaluation_truth/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}

SIMULATOR_MODES = {
    "proxy": "dual_compare_proxy_controls",
    "scenario": "dual_compare_real_controls",
    "pypsa": "dual_compare_pypsa_controls",
}

PREPROBE_MODES = {
    "full": "full",
    "context": "context_only",
    "none": "none",
}

INVARIANT_IGNORE_KEYS = {
    "run_id",
    "start_timestamp",
    "asset_simulator_mode",
    "preprobe_mode",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or check the S06 equal ablation pilot matrix.")
    parser.add_argument("--check-only", action="store_true", help="Only run matrix sanity checks against existing configs.")
    args = parser.parse_args()

    if args.check_only:
        configs = _read_config_list(ROOT / "all.txt")
        _sanity_check(configs)
        print(json.dumps({"ok": True, "checked": len(configs), "config_list": str(ROOT / "all.txt")}, indent=2))
        return

    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    full_configs: list[Path] = []
    smoke_configs: list[Path] = []
    seen: set[str] = set()

    for simulator_slug, simulator_mode in SIMULATOR_MODES.items():
        for preprobe_slug, preprobe_mode in PREPROBE_MODES.items():
            for window_slug, timestamp in WINDOWS.items():
                run_id = f"tsa-s06-{simulator_slug}-{preprobe_slug}-{window_slug}-seed42-q32"
                path = ROOT / "full" / simulator_slug / preprobe_slug / f"{run_id}.yaml"
                payload = _config(
                    run_id=run_id,
                    timestamp=timestamp,
                    ticks=24,
                    simulator_mode=simulator_mode,
                    preprobe_mode=preprobe_mode,
                )
                _write_config(path, payload, seen)
                full_configs.append(path)

            run_id = f"smoke-tsa-s06-{simulator_slug}-{preprobe_slug}-apr02-0530-2-seed42-q32"
            path = ROOT / "smoke" / simulator_slug / preprobe_slug / f"{run_id}.yaml"
            payload = _config(
                run_id=run_id,
                timestamp=WINDOWS["apr02-0530"],
                ticks=2,
                simulator_mode=simulator_mode,
                preprobe_mode=preprobe_mode,
            )
            _write_config(path, payload, seen)
            smoke_configs.append(path)

    _assert_no_existing_outputs({path.stem for path in full_configs + smoke_configs})
    _sanity_check(full_configs)
    _write_lists(full_configs=full_configs, smoke_configs=smoke_configs)
    _write_manifest(full_configs=full_configs, smoke_configs=smoke_configs)
    _write_runbook()
    _write_launcher()

    print(
        json.dumps(
            {
                "ok": True,
                "full_run_count": len(full_configs),
                "smoke_run_count": len(smoke_configs),
                "config_root": str(ROOT),
                "run_root": str(RUN_ROOT),
            },
            indent=2,
        )
    )


def _config(
    *,
    run_id: str,
    timestamp: str,
    ticks: int,
    simulator_mode: str,
    preprobe_mode: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": 42,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": 6,
        "ticks": ticks,
        "start_timestamp": timestamp,
        "forecaster_backend": "f8",
        "forecaster_routing_mode": "run_level",
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "preprobe_mode": preprobe_mode,
        "objective": "bid_seeking",
        "final_bid_guard": "simulator_exact_match",
        "safety_toolset": "full",
        "ablation_strategy": "comm_broadcast_digest_priority_calibration",
        "persona_profile": "action_core_8",
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "asset_simulator_v1",
        "asset_simulator_mode": simulator_mode,
        "asset_proxy_style": "market",
        "candidate_sizing_mode": "medium",
        "candidate_sizing_cap_fraction": 1.0,
        "candidate_sizing_min_mwh": 0.25,
        "candidate_sizing_max_candidates": 8,
        "max_tool_rounds": 6,
        "simulator_max_concurrency": 8,
        "data_start": "2026-04-01T00:00:00Z",
        "data_end": "2026-05-01T00:00:00Z",
        "context_dataset_dir": CONTEXT_DIR,
        "data_cache_dir": f"{CONTEXT_DIR}/source_cache",
        "default_lookback_hours": 24,
        "cache_refresh": False,
        "output_dir": str(RUN_ROOT),
        "memory_enabled": False,
        "memory_bank_path": MEMORY_BANK,
        "memory_max_items_per_agent": 5,
        "memory_max_prompt_chars": 2400,
        "reviewer_mode": "code_only",
        "llm": {
            "enabled": True,
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "api_key": "heimdall-local",
            "model": "Qwen/Qwen3-32B",
            "temperature": 0.2,
            "max_tokens": 640,
            "timeout_seconds": 180,
            "max_concurrency": 8,
            "per_endpoint_max_concurrency": 4,
        },
    }


def _write_config(path: Path, payload: dict[str, Any], seen: set[str]) -> None:
    run_id = str(payload["run_id"])
    if run_id in seen:
        raise RuntimeError(f"duplicate run_id: {run_id}")
    seen.add(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_lists(*, full_configs: list[Path], smoke_configs: list[Path]) -> None:
    (ROOT / "all.txt").write_text("".join(f"{path}\n" for path in full_configs), encoding="utf-8")
    (ROOT / "smoke.txt").write_text("".join(f"{path}\n" for path in smoke_configs), encoding="utf-8")


def _write_manifest(*, full_configs: list[Path], smoke_configs: list[Path]) -> None:
    payload = {
        "matrix": "thesis-s06-equal-ablation",
        "generated_at": "2026-05-19",
        "full_run_count": len(full_configs),
        "smoke_run_count": len(smoke_configs),
        "seed": 42,
        "windows": WINDOWS,
        "simulator_modes": SIMULATOR_MODES,
        "preprobe_modes": PREPROBE_MODES,
        "fixed": {
            "agent_count": 6,
            "persona_profile": "action_core_8",
            "forecaster_backend": "f8",
            "tool_policy": "asset_simulator_v1",
            "candidate_sizing_mode": "medium",
            "llm_model": "Qwen/Qwen3-32B",
        },
        "config_list": str(ROOT / "all.txt"),
        "smoke_list": str(ROOT / "smoke.txt"),
        "run_root": str(RUN_ROOT),
        "truth_dir": TRUTH_DIR,
        "sanity_check": "Full configs differ only by run_id, start_timestamp, asset_simulator_mode, and preprobe_mode.",
    }
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_runbook() -> None:
    (ROOT / "RUNBOOK.md").write_text(RUNBOOK, encoding="utf-8")


def _write_launcher() -> None:
    script = ROOT / "run_thesis_s06_equal_ablation.sh"
    script.write_text(LAUNCHER, encoding="utf-8")
    script.chmod(0o755)


def _read_config_list(path: Path) -> list[Path]:
    if not path.exists():
        raise RuntimeError(f"missing config list: {path}")
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sanity_check(config_paths: list[Path]) -> None:
    if len(config_paths) != 27:
        raise RuntimeError(f"expected 27 full configs, found {len(config_paths)}")
    payloads = [_load_config(path) for path in config_paths]
    run_ids = [payload["run_id"] for payload in payloads]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError("duplicate run_id in full matrix")
    axes = {
        "asset_simulator_mode": {payload["asset_simulator_mode"] for payload in payloads},
        "preprobe_mode": {payload["preprobe_mode"] for payload in payloads},
        "start_timestamp": {payload["start_timestamp"] for payload in payloads},
    }
    if axes["asset_simulator_mode"] != set(SIMULATOR_MODES.values()):
        raise RuntimeError(f"bad simulator axis: {axes['asset_simulator_mode']}")
    if axes["preprobe_mode"] != set(PREPROBE_MODES.values()):
        raise RuntimeError(f"bad preprobe axis: {axes['preprobe_mode']}")
    if axes["start_timestamp"] != set(WINDOWS.values()):
        raise RuntimeError(f"bad window axis: {axes['start_timestamp']}")
    base = _invariant_payload(payloads[0])
    for path, payload in zip(config_paths, payloads, strict=True):
        if _invariant_payload(payload) != base:
            raise RuntimeError(f"non-axis config drift in {path}")


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid config payload: {path}")
    if payload.get("tool_policy") != "asset_simulator_v1":
        raise RuntimeError(f"tool_policy must be asset_simulator_v1: {path}")
    return payload


def _invariant_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(payload)
    for key in INVARIANT_IGNORE_KEYS:
        normalized.pop(key, None)
    return normalized


def _assert_no_existing_outputs(run_ids: set[str]) -> None:
    collisions: list[str] = []
    for run_id in sorted(run_ids):
        for path in (RUN_ROOT / run_id, Path("evaluations") / run_id):
            if path.exists():
                collisions.append(str(path))
    if collisions:
        raise RuntimeError("refusing to generate configs for existing outputs:\n" + "\n".join(collisions))


RUNBOOK = """# Thesis S06 Equal Ablation 2026-05-19

S06-only thesis pilot with one comparison seed (`42`), three simulator-control levels, three tool-autonomy levels, and three 24-tick opportunity windows.

## Matrix

- 27 full runs: `proxy`, `scenario`, `pypsa` x `full`, `context_only`, `none` x Apr02/Apr09/Apr13.
- 9 smoke runs: one 2-tick Apr02 smoke for every simulator x tool-autonomy pair.
- Fixed society: `agent_count: 6`, `persona_profile: action_core_8`, `forecaster_backend: f8`.
- Fixed guard: `tool_policy: asset_simulator_v1`, `final_bid_guard: simulator_exact_match`, `safety_toolset: full`.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_thesis_s06_equal_ablation.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/thesis-s06-equal-ablation-20260519/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/thesis-s06-equal-ablation-20260519/all.txt
```

## Launch

Start both vLLM endpoints for `Qwen/Qwen3-32B`, then run:

```bash
cd /home/ucloud/heimdall
tmux new-session -d -s heimdall-tsa-s06 "bash ai-society/configs/thesis-s06-equal-ablation-20260519/run_thesis_s06_equal_ablation.sh > ai-society/configs/thesis-s06-equal-ablation-20260519/chain.log 2>&1"
```

## Monitor

```bash
tail -f ai-society/configs/thesis-s06-equal-ablation-20260519/chain.log
cat ai-society/configs/thesis-s06-equal-ablation-20260519/latest-smoke-log-dir.txt
cat ai-society/configs/thesis-s06-equal-ablation-20260519/latest-log-dir.txt
tail -f "$(cat ai-society/configs/thesis-s06-equal-ablation-20260519/latest-log-dir.txt)/sequential.stdout.log"
```

The stage runner validates simulator gating and preprobe provenance before continuing.
"""


LAUNCHER = """#!/usr/bin/env bash
set -euo pipefail

cd /home/ucloud/heimdall
export PYTHONPATH=.

ROOT="ai-society/configs/thesis-s06-equal-ablation-20260519"
RUN_ROOT="ai-society/runs/thesis-s06-equal-ablation-20260519"
STAGE="thesis-s06-equal-ablation"
BASE_URL="http://127.0.0.1:8000/v1"

validate_list() {
  local list_path="$1"
  while read -r cfg; do
    [ -z "$cfg" ] && continue
    uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
  done < "$list_path"
}

run_stage() {
  local config_list="$1"
  local log_dir="$2"
  uv run python ai-society/run_market_intelligence_stage.py \\
    --stage "$STAGE" \\
    --gpu gpu0 \\
    --base-url "$BASE_URL" \\
    --config-list "$config_list" \\
    --log-dir "$log_dir" \\
    > "$log_dir/sequential.stdout.log" 2>&1
}

uv run python tools/experiments/generate_thesis_s06_equal_ablation.py --check-only
validate_list "$ROOT/smoke.txt"
validate_list "$ROOT/all.txt"

smoke_log_dir="$RUN_ROOT/logs-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$smoke_log_dir"
printf '%s\\n' "$smoke_log_dir" > "$ROOT/latest-smoke-log-dir.txt"
run_stage "$ROOT/smoke.txt" "$smoke_log_dir"

full_log_dir="$RUN_ROOT/logs-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$full_log_dir"
printf '%s\\n' "$full_log_dir" > "$ROOT/latest-log-dir.txt"
run_stage "$ROOT/all.txt" "$full_log_dir"
"""


if __name__ == "__main__":
    main()
