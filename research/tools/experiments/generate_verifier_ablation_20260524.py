"""Generate the verifier vs verifierless ablation matrix (robust, multi-seed, multi-window).

Clean toggle of the two-stage verifier (physical + conformal worst-case-profit) holding the
proposer, society, tools, forecaster, and market window constant. Four variants:

  - deterministic        : chooser=deterministic_best_accepted, final_bid_guard=simulator_exact_match,
                           llm off.  Verifier ON (selection + final guard); a well-formed proposer that
                           rarely needs the floor -> the always-safe reference.
  - guarded              : chooser=llm, final_bid_guard=simulator_exact_match, tools on.  Verifier ON.
  - shadow-toolvisible    : chooser=llm, final_bid_guard=schema_only_shadow,   tools on.  Verifier OFF
                           (the bid is shadow-scored so we still record what the verifier WOULD block).
  - shadow-contextonly    : chooser=llm, final_bid_guard=schema_only_shadow,   tools off (context_only).
                           Verifier OFF + ungrounded -> the regime where the floor matters most.

The clean verifier contribution is guarded vs shadow-toolvisible (tools held constant, only the gate
differs). shadow-contextonly shows the value of the gate when grounding degrades.

Matrix: 1 society (s06-actioncore) x 4 variants x 5 windows x 5 frozen seeds = 100 runs.

Usage:
    PYTHONPATH=. uv run python tools/experiments/generate_verifier_ablation_20260524.py
    PYTHONPATH=. uv run python tools/experiments/generate_verifier_ablation_20260524.py --check-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

MATRIX = "verifier-ablation-20260524"
ROOT = Path(f"ai-society/configs/{MATRIX}")
RUN_ROOT = Path(f"ai-society/runs/{MATRIX}")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"
MODEL = "Qwen/Qwen3-32B"

# Frozen seeds (project hard constraint).
SEEDS = [13, 42, 137, 1729, 31415]

# Five windows spanning the April 2026 evaluation month (different volatility regimes).
WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
    "apr17-1900": "2026-04-17T19:00:00Z",
    "apr28-1900": "2026-04-28T19:00:00Z",
}

SOCIETY = {"slug": "s06-actioncore", "agent_count": 6, "persona_profile": "action_core_8"}

VARIANTS = {
    "deterministic": {
        "chooser_mode": "deterministic_best_accepted",
        "final_bid_guard": "simulator_exact_match",
        "safety_toolset": "full",
        "llm_enabled": False,
        "temperature": 0.0,
        "max_tokens": 512,
    },
    "guarded": {
        "chooser_mode": "llm",
        "final_bid_guard": "simulator_exact_match",
        "safety_toolset": "full",
        "llm_enabled": True,
        "temperature": 0.2,
        "max_tokens": 640,
    },
    "shadow-toolvisible": {
        "chooser_mode": "llm",
        "final_bid_guard": "schema_only_shadow",
        "safety_toolset": "full",
        "llm_enabled": True,
        "temperature": 0.2,
        "max_tokens": 640,
    },
    "shadow-contextonly": {
        "chooser_mode": "llm",
        "final_bid_guard": "schema_only_shadow",
        "safety_toolset": "context_only",
        "llm_enabled": True,
        "temperature": 0.2,
        "max_tokens": 640,
    },
}


def _config(*, run_id: str, variant_slug: str, timestamp: str, seed: int, ticks: int) -> dict[str, Any]:
    v = VARIANTS[variant_slug]
    return {
        "run_id": run_id,
        "seed": seed,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": SOCIETY["agent_count"],
        "ticks": ticks,
        "start_timestamp": timestamp,
        "forecaster_backend": "f8",
        "forecaster_routing_mode": "persona",
        "chooser_mode": v["chooser_mode"],
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "preprobe_mode": "full",
        "objective": "bid_seeking",
        "final_bid_guard": v["final_bid_guard"],
        "safety_toolset": v["safety_toolset"],
        "ablation_strategy": "comm_broadcast_digest_priority_calibration",
        "persona_profile": SOCIETY["persona_profile"],
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "asset_simulator_v1",
        "asset_simulator_mode": "scenario_envelope",
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
            "enabled": v["llm_enabled"],
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "api_key": "heimdall-local",
            "model": MODEL,
            "temperature": v["temperature"],
            "max_tokens": v["max_tokens"],
            "timeout_seconds": 180,
            "max_concurrency": 12,
            "per_endpoint_max_concurrency": 6,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    full: list[Path] = []
    det: list[Path] = []
    llm: list[Path] = []
    seen: set[str] = set()
    for seed in SEEDS:
        for window_slug, ts in WINDOWS.items():
            for variant_slug in VARIANTS:
                rid = f"vab-{SOCIETY['slug']}-{variant_slug}-{window_slug}-seed{seed}-24-q32"
                if rid in seen:
                    raise RuntimeError(f"duplicate run_id {rid}")
                seen.add(rid)
                path = ROOT / variant_slug / f"{rid}.yaml"
                if not args.check_only:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        yaml.safe_dump(
                            _config(run_id=rid, variant_slug=variant_slug, timestamp=ts, seed=seed, ticks=24),
                            sort_keys=False,
                        ),
                        encoding="utf-8",
                    )
                full.append(path)
                (det if variant_slug == "deterministic" else llm).append(path)

    if not args.check_only:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "all.txt").write_text("".join(f"{p}\n" for p in full))
        (ROOT / "det.txt").write_text("".join(f"{p}\n" for p in det))
        (ROOT / "llm.txt").write_text("".join(f"{p}\n" for p in llm))

    print(
        json.dumps(
            {
                "ok": True,
                "total": len(full),
                "det_runs": len(det),
                "llm_runs": len(llm),
                "seeds": SEEDS,
                "windows": list(WINDOWS),
                "variants": list(VARIANTS),
                "config_root": str(ROOT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
