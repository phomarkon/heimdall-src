from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/verifierless-baseline-20260519")
RUN_ROOT = Path("ai-society/runs/verifierless-baseline-20260519")
CONTEXT_DIR = "data/cache/real_context/april_2026"
TRUTH_DIR = "data/cache/evaluation_truth/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"
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
        "forecaster_routing_mode": "run_level",
        "max_concurrency": 8,
        "per_endpoint_max_concurrency": 4,
    },
    "mixed20": {
        "agent_count": 20,
        "persona_profile": "mixed_expert_20_sideaware",
        "forecaster_routing_mode": "persona",
        "max_concurrency": 16,
        "per_endpoint_max_concurrency": 8,
    },
}

VARIANTS = {
    "guarded": {
        "objective": "bid_seeking",
        "final_bid_guard": "simulator_exact_match",
        "safety_toolset": "full",
        "preprobe_mode": "full",
    },
    "shadow-toolvisible": {
        "objective": "bid_seeking",
        "final_bid_guard": "schema_only_shadow",
        "safety_toolset": "full",
        "preprobe_mode": "full",
    },
    "shadow-contextonly": {
        "objective": "unverified_bid_seeking",
        "final_bid_guard": "schema_only_shadow",
        "safety_toolset": "context_only",
        "preprobe_mode": "context_only",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or check the verifierless baseline matrix.")
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

    for window_slug, timestamp in WINDOWS.items():
        for variant_slug in ("shadow-toolvisible", "shadow-contextonly"):
            run_id = f"vlb-s06-{variant_slug}-{window_slug}-seed42-q32"
            path = ROOT / "full" / "s06-actioncore" / variant_slug / f"{run_id}.yaml"
            _write_config(path, _config(run_id, "s06-actioncore", variant_slug, timestamp, 24), seen)
            full_configs.append(path)

        for variant_slug in ("guarded", "shadow-toolvisible", "shadow-contextonly"):
            run_id = f"vlb-mixed20-{variant_slug}-{window_slug}-seed42-q32"
            path = ROOT / "full" / "mixed20" / variant_slug / f"{run_id}.yaml"
            _write_config(path, _config(run_id, "mixed20", variant_slug, timestamp, 24), seen)
            full_configs.append(path)

    for society_slug, variant_slug in [
        ("s06-actioncore", "shadow-toolvisible"),
        ("s06-actioncore", "shadow-contextonly"),
        ("mixed20", "guarded"),
        ("mixed20", "shadow-toolvisible"),
        ("mixed20", "shadow-contextonly"),
    ]:
        run_id = f"smoke-vlb-{society_slug}-{variant_slug}-apr02-0530-2-seed42-q32"
        path = ROOT / "smoke" / society_slug / variant_slug / f"{run_id}.yaml"
        _write_config(path, _config(run_id, society_slug, variant_slug, WINDOWS["apr02-0530"], 2), seen)
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


def _config(run_id: str, society_slug: str, variant_slug: str, timestamp: str, ticks: int) -> dict[str, Any]:
    society = SOCIETIES[society_slug]
    variant = VARIANTS[variant_slug]
    payload: dict[str, Any] = {
        "run_id": run_id,
        "seed": 42,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": society["agent_count"],
        "ticks": ticks,
        "start_timestamp": timestamp,
        "forecaster_backend": "f8",
        "forecaster_routing_mode": society["forecaster_routing_mode"],
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "objective": variant["objective"],
        "final_bid_guard": variant["final_bid_guard"],
        "safety_toolset": variant["safety_toolset"],
        "preprobe_mode": variant["preprobe_mode"],
        "ablation_strategy": "comm_broadcast_digest_priority_calibration",
        "persona_profile": society["persona_profile"],
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "asset_simulator_v1",
        "asset_simulator_mode": "dual_compare_real_controls",
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
            "model": MODEL,
            "temperature": 0.2,
            "max_tokens": 640,
            "timeout_seconds": 180,
            "max_concurrency": society["max_concurrency"],
            "per_endpoint_max_concurrency": society["per_endpoint_max_concurrency"],
        },
    }
    return payload


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
        "matrix": "verifierless-baseline-20260519",
        "generated_at": "2026-05-19",
        "full_run_count": len(full_configs),
        "smoke_run_count": len(smoke_configs),
        "seed": 42,
        "forecaster_backend": "f8",
        "windows": WINDOWS,
        "run_root": str(RUN_ROOT),
        "truth_dir": TRUTH_DIR,
        "config_list": str(ROOT / "all.txt"),
        "smoke_list": str(ROOT / "smoke.txt"),
        "s06_guarded_comparator_source": "thesis-s06-equal-ablation-20260519 scenario/full medium cells",
        "variants": VARIANTS,
        "full_runs": [_run_manifest_row(path) for path in full_configs],
    }
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_manifest_row(path: Path) -> dict[str, str]:
    parts = path.parts
    return {
        "run_id": path.stem,
        "config": str(path),
        "society": parts[len(ROOT.parts) + 1],
        "variant": parts[len(ROOT.parts) + 2],
    }


def _write_runbook() -> None:
    (ROOT / "RUNBOOK.md").write_text(RUNBOOK, encoding="utf-8")


def _read_config_list(path: Path) -> list[Path]:
    if not path.exists():
        raise RuntimeError(f"missing config list: {path}")
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sanity_check(full_configs: list[Path], smoke_configs: list[Path]) -> None:
    if len(full_configs) != 15:
        raise RuntimeError(f"expected 15 full configs, found {len(full_configs)}")
    if len(smoke_configs) != 5:
        raise RuntimeError(f"expected 5 smoke configs, found {len(smoke_configs)}")
    payloads = [_load_config(path) for path in full_configs]
    run_ids = [str(payload["run_id"]) for payload in payloads]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError("duplicate full run_id")
    s06 = [payload for payload in payloads if str(payload["run_id"]).startswith("vlb-s06-")]
    mixed20 = [payload for payload in payloads if str(payload["run_id"]).startswith("vlb-mixed20-")]
    if len(s06) != 6 or len(mixed20) != 9:
        raise RuntimeError(f"bad society split: s06={len(s06)} mixed20={len(mixed20)}")
    for payload in payloads:
        if payload.get("forecaster_backend") != "f8" or payload.get("seed") != 42:
            raise RuntimeError(f"bad fixed forecaster/seed in {payload.get('run_id')}")
        if payload.get("asset_simulator_mode") != "dual_compare_real_controls":
            raise RuntimeError(f"bad simulator mode in {payload.get('run_id')}")
        if "shadow-contextonly" in str(payload["run_id"]):
            _expect_subset(
                payload,
                {
                    "final_bid_guard": "schema_only_shadow",
                    "safety_toolset": "context_only",
                    "preprobe_mode": "context_only",
                    "objective": "unverified_bid_seeking",
                },
            )
        if "shadow-toolvisible" in str(payload["run_id"]):
            _expect_subset(
                payload,
                {
                    "final_bid_guard": "schema_only_shadow",
                    "safety_toolset": "full",
                    "preprobe_mode": "full",
                    "objective": "bid_seeking",
                },
            )
        if "-guarded-" in str(payload["run_id"]):
            _expect_subset(
                payload,
                {
                    "final_bid_guard": "simulator_exact_match",
                    "safety_toolset": "full",
                    "preprobe_mode": "full",
                    "objective": "bid_seeking",
                },
            )
    _assert_axis_only_drift(s06)


def _assert_axis_only_drift(payloads: list[dict[str, Any]]) -> None:
    ignore = {"run_id", "start_timestamp", "final_bid_guard", "safety_toolset", "preprobe_mode", "objective"}
    base = _without_keys(payloads[0], ignore)
    for payload in payloads[1:]:
        if _without_keys(payload, ignore) != base:
            raise RuntimeError(f"unexpected S06 config drift in {payload['run_id']}")


def _expect_subset(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"{payload.get('run_id')} has {key}={payload.get(key)!r}, expected {value!r}")


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
        raise RuntimeError("refusing to generate duplicate verifierless outputs:\n" + "\n".join(collisions))


RUNBOOK = """# Verifierless Baseline 2026-05-19

This matrix launches after the current `thesis-s06-equal-ablation-20260519` TSA run completes cleanly.

## Matrix

- 15 full 24-tick runs.
- S06 action-core: `shadow-toolvisible` and `shadow-contextonly` on Apr02, Apr09, Apr13.
- Mixed20 side-aware: guarded real-controls, `shadow-toolvisible`, and `shadow-contextonly` on Apr02, Apr09, Apr13.
- Fixed seed and forecaster: `seed: 42`, `forecaster_backend: f8`.

## Safety Semantics

- `shadow-toolvisible`: simulator and feasibility tools are available, but final exact-match gating is disabled and every bid is shadow-scored after the decision.
- `shadow-contextonly`: pre-decision simulator, feasibility, candidate, ranker, and candidate-guidance tools are hidden. The prompt uses `unverified_bid_seeking`; submitted bids are shadow-scored only after the decision.
- S06 guarded comparators are the already-running TSA `scenario/full/medium` cells.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_verifierless_baseline.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/verifierless-baseline-20260519/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/verifierless-baseline-20260519/all.txt
```

## Compare

```bash
PYTHONPATH=. uv run python tools/evaluation/compare_verifierless_baseline.py
```
"""


if __name__ == "__main__":
    main()
