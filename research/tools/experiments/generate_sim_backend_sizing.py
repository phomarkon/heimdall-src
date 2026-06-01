from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/sim-backend-sizing-20260519")
RUN_ROOT = Path("ai-society/runs/sim-backend-sizing-20260519")
CONTEXT_DIR = "data/cache/real_context/april_2026"
TRUTH_DIR = "data/cache/evaluation_truth/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"
MODEL = "Qwen/Qwen3-32B"

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}

BACKENDS = {
    "scenario": "dual_compare_real_controls",
    "pypsa": "dual_compare_pypsa_controls",
}

SIZING = {
    "current": {
        "candidate_sizing_mode": "current",
        "candidate_sizing_cap_fraction": 1.0,
        "candidate_sizing_min_mwh": 0.25,
        "candidate_sizing_max_candidates": 8,
    },
    "large": {
        "candidate_sizing_mode": "large",
        "candidate_sizing_cap_fraction": 1.0,
        "candidate_sizing_min_mwh": 0.25,
        "candidate_sizing_max_candidates": 8,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or check the simulator backend sizing matrix.")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        full = _read_config_list(ROOT / "all.txt")
        smoke = _read_config_list(ROOT / "smoke.txt")
        _sanity_check(full, smoke)
        print(json.dumps({"ok": True, "full": len(full), "smoke": len(smoke), "config_root": str(ROOT)}, indent=2))
        return

    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    full_configs: list[Path] = []
    smoke_configs: list[Path] = []
    seen: set[str] = set()

    for backend_slug, backend_mode in BACKENDS.items():
        for sizing_slug in SIZING:
            for window_slug, timestamp in WINDOWS.items():
                run_id = f"sbs-s06-{backend_slug}-{sizing_slug}-{window_slug}-seed42-q32"
                path = ROOT / "full" / backend_slug / sizing_slug / f"{run_id}.yaml"
                _write_config(path, _config(run_id, backend_mode, sizing_slug, timestamp, 24), seen)
                full_configs.append(path)

            run_id = f"smoke-sbs-s06-{backend_slug}-{sizing_slug}-apr02-0530-2-seed42-q32"
            path = ROOT / "smoke" / backend_slug / sizing_slug / f"{run_id}.yaml"
            _write_config(path, _config(run_id, backend_mode, sizing_slug, WINDOWS["apr02-0530"], 2), seen)
            smoke_configs.append(path)

    _assert_no_existing_outputs({path.stem for path in full_configs + smoke_configs})
    _sanity_check(full_configs, smoke_configs)
    _write_lists(full_configs, smoke_configs)
    _write_manifest(full_configs, smoke_configs)
    _write_runbook()

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


def _config(run_id: str, backend_mode: str, sizing_slug: str, timestamp: str, ticks: int) -> dict[str, Any]:
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
        "preprobe_mode": "full",
        "objective": "bid_seeking",
        "final_bid_guard": "simulator_exact_match",
        "safety_toolset": "full",
        "ablation_strategy": "comm_broadcast_digest_priority_calibration",
        "persona_profile": "action_core_8",
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "asset_simulator_v1",
        "asset_simulator_mode": backend_mode,
        "asset_proxy_style": "market",
        **SIZING[sizing_slug],
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
            "model": MODEL,
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


def _write_lists(full_configs: list[Path], smoke_configs: list[Path]) -> None:
    (ROOT / "all.txt").write_text("".join(f"{path}\n" for path in full_configs), encoding="utf-8")
    (ROOT / "smoke.txt").write_text("".join(f"{path}\n" for path in smoke_configs), encoding="utf-8")


def _write_manifest(full_configs: list[Path], smoke_configs: list[Path]) -> None:
    payload = {
        "matrix": "sim-backend-sizing-20260519",
        "generated_at": "2026-05-19",
        "full_run_count": len(full_configs),
        "smoke_run_count": len(smoke_configs),
        "seed": 42,
        "forecaster_backend": "f8",
        "windows": WINDOWS,
        "backends": BACKENDS,
        "new_sizing_arms": list(SIZING),
        "medium_comparator_source": "thesis-s06-equal-ablation-20260519 scenario/full and pypsa/full medium cells",
        "run_root": str(RUN_ROOT),
        "truth_dir": TRUTH_DIR,
        "config_list": str(ROOT / "all.txt"),
        "smoke_list": str(ROOT / "smoke.txt"),
        "full_runs": [_run_manifest_row(path) for path in full_configs],
    }
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_manifest_row(path: Path) -> dict[str, str]:
    parts = path.parts
    return {
        "run_id": path.stem,
        "config": str(path),
        "backend": parts[len(ROOT.parts) + 1],
        "sizing": parts[len(ROOT.parts) + 2],
    }


def _write_runbook() -> None:
    (ROOT / "RUNBOOK.md").write_text(RUNBOOK, encoding="utf-8")


def _read_config_list(path: Path) -> list[Path]:
    if not path.exists():
        raise RuntimeError(f"missing config list: {path}")
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sanity_check(full_configs: list[Path], smoke_configs: list[Path]) -> None:
    if len(full_configs) != 12:
        raise RuntimeError(f"expected 12 full configs, found {len(full_configs)}")
    if len(smoke_configs) != 4:
        raise RuntimeError(f"expected 4 smoke configs, found {len(smoke_configs)}")
    payloads = [_load_config(path) for path in full_configs]
    run_ids = [str(payload["run_id"]) for payload in payloads]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError("duplicate full run_id")
    if {payload["asset_simulator_mode"] for payload in payloads} != set(BACKENDS.values()):
        raise RuntimeError("bad backend axis")
    if {payload["candidate_sizing_mode"] for payload in payloads} != set(SIZING):
        raise RuntimeError("bad sizing axis")
    if {payload["start_timestamp"] for payload in payloads} != set(WINDOWS.values()):
        raise RuntimeError("bad window axis")
    base = _without_keys(payloads[0], {"run_id", "start_timestamp", "asset_simulator_mode", "candidate_sizing_mode"})
    for payload in payloads[1:]:
        comparable = _without_keys(payload, {"run_id", "start_timestamp", "asset_simulator_mode", "candidate_sizing_mode"})
        if comparable != base:
            raise RuntimeError(f"unexpected non-axis drift in {payload['run_id']}")


def _without_keys(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    copied = deepcopy(payload)
    for key in keys:
        copied.pop(key, None)
    return copied


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid config payload: {path}")
    return payload


def _assert_no_existing_outputs(run_ids: set[str]) -> None:
    collisions = []
    for run_id in sorted(run_ids):
        for path in (RUN_ROOT / run_id, Path("evaluations") / run_id):
            if path.exists():
                collisions.append(str(path))
    if collisions:
        raise RuntimeError("refusing to generate duplicate sim-backend-sizing outputs:\n" + "\n".join(collisions))


RUNBOOK = """# Simulator Backend Sizing 2026-05-19

This guarded S06 matrix runs after the verifierless baseline matrix.

## Matrix

- 12 new full 24-tick S06 runs.
- Backends: `dual_compare_real_controls`, `dual_compare_pypsa_controls`.
- New sizing arms: `current`, `large`.
- Existing medium comparators are reused from the current TSA `scenario/full` and `pypsa/full` cells.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_sim_backend_sizing.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/sim-backend-sizing-20260519/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/sim-backend-sizing-20260519/all.txt
```

## Compare

```bash
PYTHONPATH=. uv run python tools/evaluation/compare_sim_backend_sizing.py
```
"""


if __name__ == "__main__":
    main()
