"""Generate the llm-value-allarch-20260523 matrix (first grounded LLM-value test).

Society: all_archetypes_v1 = one focal verifier-guarded P2H agent (50 MW, real PyPSA-Eur-Sec
spec) + one each of EV/WIND/GENERATOR/RETAILER/RENEWABLES. P2H-focal capacity capture
(tools/evaluation/rescore_runs.py) is the grounded headline metric; raw realized profit is the
secondary.

One mechanism moves per arm so a difference is attributable:
  det      deterministic_best_accepted (verifier-gated)      -- control floor
  selector LLM selects over the same pre-ranked menu         -- selection value (selector trap)
  cp12     LLM proposes its own side/qty/price (guard off)   -- generative agency (incl. sizing)
  cp13     generative + frontier-refine reprompt (guarded)   -- aggressive feasible frontier

Held constant: society, sizing, verifier floor (tau=0), forecaster, context. Start small:
2 active windows x seed 42 (fast first read); expand seeds after inspecting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

MATRIX = "llm-value-allarch-20260523"
ROOT = Path(f"ai-society/configs/{MATRIX}")
RUN_ROOT = Path(f"ai-society/runs/{MATRIX}")
CONTEXT_DIR = "data/cache/real_context/april_2026"
TRUTH_DIR = "data/cache/evaluation_truth/april_2026"
MODEL = "Qwen/Qwen3-32B"
ENDPOINTS = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1"]

SEEDS = [42]
SMOKE_SEED = 42
WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
}
AGENT_COUNT = 6
PERSONA_PROFILE = "all_archetypes_v1"

# Per-arm knobs (only the value mechanism moves). cp12/cp13 are the agency arms unblocked by the
# runner autonomy fix (forced first-round tool call when no menu is seeded).
MODES: dict[str, dict[str, Any]] = {
    "det": {
        "chooser_mode": "deterministic_best_accepted",
        "llm_enabled": False,
        "preprobe_mode": "full",
        "final_bid_guard": "schema_only_shadow",  # moot: no LLM bid
        "ablation_strategy": "baseline",
        "temperature": 0.0,
        "max_tokens": 512,
    },
    "selector": {
        "chooser_mode": "llm",
        "llm_enabled": True,
        "preprobe_mode": "full",
        "final_bid_guard": "simulator_exact_match",
        "ablation_strategy": "baseline",
        "temperature": 0.2,
        "max_tokens": 1024,
    },
    "cp12": {
        "chooser_mode": "llm",
        "llm_enabled": True,
        "preprobe_mode": "context_only",
        "final_bid_guard": "schema_only_shadow",  # LLM bid judged by verifier+downside, not snapped
        "ablation_strategy": "cp12_llm_suggest_plus_code_ladder",
        "temperature": 0.2,
        "max_tokens": 1024,
    },
    "cp13": {
        "chooser_mode": "llm",
        "llm_enabled": True,
        "preprobe_mode": "context_only",
        "final_bid_guard": "simulator_exact_match",  # triggers the frontier-refine reprompt loop
        "ablation_strategy": "cp13_llm_probe_refine_frontier",
        "temperature": 0.2,
        "max_tokens": 1024,
    },
}

CONSTANTS = {
    "zone": "DK1",
    "agent_count": AGENT_COUNT,
    "forecaster_backend": "f8",
    "forecaster_routing_mode": "persona",
    "verifier_mode": "simulator",
    "verifier_tau_eur": 0.0,  # strict floor, constant across arms
    "market_context": "real",
    "tool_mode": "openai_tools",
    "objective": "bid_seeking",
    "safety_toolset": "full",
    "persona_profile": PERSONA_PROFILE,
    "scenario_id": "p2h_dk1_pypsa",
    "tool_policy": "asset_simulator_v1",
    "asset_simulator_mode": "scenario_envelope",
    "asset_proxy_style": "market",
    "candidate_sizing_mode": "large",
    "candidate_sizing_cap_fraction": 1.0,
    "candidate_sizing_min_mwh": 0.25,
    "candidate_sizing_max_candidates": 8,
    "max_tool_rounds": 6,
    "simulator_max_concurrency": 12,
    "data_start": "2026-04-01T00:00:00Z",
    "data_end": "2026-05-01T00:00:00Z",
    "context_dataset_dir": CONTEXT_DIR,
    "data_cache_dir": f"{CONTEXT_DIR}/source_cache",
    "default_lookback_hours": 24,
    "cache_refresh": False,
    "memory_enabled": False,
    "reviewer_mode": "code_only",
}

EXPECTED_FULL = len(MODES) * len(WINDOWS) * len(SEEDS)
# Smoke: the two agency arms only (this is what the autonomy gate must verify).
SMOKE_MODES = ["cp12", "cp13"]


def _config(*, run_id: str, mode_slug: str, timestamp: str, ticks: int, seed: int) -> dict[str, Any]:
    mode = MODES[mode_slug]
    payload: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "forecaster_seed": seed,
        "ticks": ticks,
        "start_timestamp": timestamp,
        "chooser_mode": mode["chooser_mode"],
        "preprobe_mode": mode["preprobe_mode"],
        "final_bid_guard": mode["final_bid_guard"],
        "ablation_strategy": mode["ablation_strategy"],
        "output_dir": str(RUN_ROOT),
        **{k: v for k, v in CONSTANTS.items()},
        "llm": {
            "enabled": mode["llm_enabled"],
            "base_urls": ENDPOINTS,
            "api_key": "heimdall-local",
            "model": MODEL,
            "temperature": mode["temperature"],
            "max_tokens": mode["max_tokens"],
            "timeout_seconds": 180,
            "max_concurrency": 18,
            "per_endpoint_max_concurrency": 6,
        },
    }
    return payload


def _write(path: Path, payload: dict[str, Any], seen: set[str]) -> None:
    rid = str(payload["run_id"])
    if rid in seen:
        raise RuntimeError(f"duplicate run_id: {rid}")
    seen.add(rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Generate {MATRIX} configs.")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        full = [p for p in (ROOT / "full").rglob("*.yaml")]
        smoke = [p for p in (ROOT / "smoke").rglob("*.yaml")]
        assert len(full) == EXPECTED_FULL, f"expected {EXPECTED_FULL} full, found {len(full)}"
        assert len(smoke) == len(SMOKE_MODES), f"expected {len(SMOKE_MODES)} smoke, found {len(smoke)}"
        print(json.dumps({"ok": True, "full": len(full), "smoke": len(smoke)}, indent=2))
        return

    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    full_paths: list[Path] = []
    smoke_paths: list[Path] = []

    for seed in SEEDS:
        for mode_slug in MODES:
            for window_slug, ts in WINDOWS.items():
                rid = f"lva-{mode_slug}-{window_slug}-seed{seed}-24-q32"
                path = ROOT / "full" / mode_slug / f"{rid}.yaml"
                _write(path, _config(run_id=rid, mode_slug=mode_slug, timestamp=ts, ticks=24, seed=seed), seen)
                full_paths.append(path)

    for mode_slug in SMOKE_MODES:
        rid = f"smoke-lva-{mode_slug}-apr02-0530-seed{SMOKE_SEED}-2-q32"
        path = ROOT / "smoke" / mode_slug / f"{rid}.yaml"
        _write(path, _config(run_id=rid, mode_slug=mode_slug, timestamp=WINDOWS["apr02-0530"], ticks=2, seed=SMOKE_SEED), seen)
        smoke_paths.append(path)

    (ROOT / "all.txt").write_text("".join(f"{p}\n" for p in full_paths), encoding="utf-8")
    (ROOT / "smoke.txt").write_text("".join(f"{p}\n" for p in smoke_paths), encoding="utf-8")
    (ROOT / "manifest.json").write_text(json.dumps({
        "matrix": MATRIX,
        "generated_at": "2026-05-23",
        "society": {"agent_count": AGENT_COUNT, "persona_profile": PERSONA_PROFILE},
        "windows": WINDOWS,
        "seeds": SEEDS,
        "arms": list(MODES),
        "primary_metric": "P2H-focal capture_capacity (rescore_runs.py) + realized_profit_eur",
        "verifier_tau_eur": 0.0,
        "endpoints": ENDPOINTS,
        "context_dir": CONTEXT_DIR,
        "truth_dir": TRUTH_DIR,
        "full_runs": len(full_paths),
        "smoke_runs": len(smoke_paths),
        "modes": MODES,
        "constants": CONSTANTS,
    }, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"ok": True, "full": len(full_paths), "smoke": len(smoke_paths), "root": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
