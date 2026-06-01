"""Strong-deterministic-baseline arms for the allarch LLM-value test.

The first matrix (llm-value-allarch-20260523) used a THIN deterministic ladder (`baseline` =
one forecast-direction candidate per tick), which made det a weak competitor (right-side
candidate only 9/144 ticks, 0 fills). This matrix adds the STRONG baseline arms over the same
constants so the comparison is fair:

  det_rich      deterministic_best_accepted + deterministic_rich ladder (dense both-sides grid)
  selector_rich LLM selects over that SAME rich pre-ranked menu (preprobe=full)

Clean reads (combined with the first matrix's cp12/cp13, which are unchanged by the ladder):
  det_rich vs selector_rich -> does LLM SELECTION beat code ranking over the same strong menu?
  det_rich vs cp12/cp13     -> does LLM adaptive probing+reasoning beat a strong det?
Same society/window/seed/tau/sizing/forecaster as the first matrix; only chooser+ladder move.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

MATRIX = "llm-value-allarch-rich-20260523"
ROOT = Path(f"ai-society/configs/{MATRIX}")
RUN_ROOT = Path(f"ai-society/runs/{MATRIX}")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MODEL = "Qwen/Qwen3-32B"
ENDPOINTS = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1"]
SEEDS = [42]
WINDOWS = {"apr02-0530": "2026-04-02T05:30:00Z", "apr09-1830": "2026-04-09T18:30:00Z"}

MODES: dict[str, dict[str, Any]] = {
    "det_rich": {"chooser_mode": "deterministic_best_accepted", "llm_enabled": False, "preprobe_mode": "full",
                 "final_bid_guard": "schema_only_shadow", "ablation_strategy": "deterministic_rich",
                 "temperature": 0.0, "max_tokens": 512},
    "selector_rich": {"chooser_mode": "llm", "llm_enabled": True, "preprobe_mode": "full",
                      "final_bid_guard": "simulator_exact_match", "ablation_strategy": "deterministic_rich",
                      "temperature": 0.2, "max_tokens": 1024},
}

CONSTANTS = {
    "zone": "DK1", "agent_count": 6, "forecaster_backend": "f8", "forecaster_routing_mode": "persona",
    "verifier_mode": "simulator", "verifier_tau_eur": 0.0, "market_context": "real", "tool_mode": "openai_tools",
    "objective": "bid_seeking", "safety_toolset": "full", "persona_profile": "all_archetypes_v1",
    "scenario_id": "p2h_dk1_pypsa", "tool_policy": "asset_simulator_v1", "asset_simulator_mode": "scenario_envelope",
    "asset_proxy_style": "market", "candidate_sizing_mode": "large", "candidate_sizing_cap_fraction": 1.0,
    "candidate_sizing_min_mwh": 0.25, "candidate_sizing_max_candidates": 8, "max_tool_rounds": 6,
    "simulator_max_concurrency": 12, "data_start": "2026-04-01T00:00:00Z", "data_end": "2026-05-01T00:00:00Z",
    "context_dataset_dir": CONTEXT_DIR, "data_cache_dir": f"{CONTEXT_DIR}/source_cache",
    "default_lookback_hours": 24, "cache_refresh": False, "memory_enabled": False, "reviewer_mode": "code_only",
}


def _config(*, run_id: str, mode_slug: str, timestamp: str, ticks: int, seed: int) -> dict[str, Any]:
    m = MODES[mode_slug]
    return {
        "run_id": run_id, "seed": seed, "forecaster_seed": seed, "ticks": ticks, "start_timestamp": timestamp,
        "chooser_mode": m["chooser_mode"], "preprobe_mode": m["preprobe_mode"], "final_bid_guard": m["final_bid_guard"],
        "ablation_strategy": m["ablation_strategy"], "output_dir": str(RUN_ROOT), **dict(CONSTANTS),
        "llm": {"enabled": m["llm_enabled"], "base_urls": ENDPOINTS, "api_key": "heimdall-local", "model": MODEL,
                "temperature": m["temperature"], "max_tokens": m["max_tokens"], "timeout_seconds": 180,
                "max_concurrency": 18, "per_endpoint_max_concurrency": 6},
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    full: list[Path] = []
    for seed in SEEDS:
        for mode_slug in MODES:
            for wslug, ts in WINDOWS.items():
                rid = f"lvar-{mode_slug}-{wslug}-seed{seed}-24-q32"
                p = ROOT / "full" / mode_slug / f"{rid}.yaml"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(yaml.safe_dump(_config(run_id=rid, mode_slug=mode_slug, timestamp=ts, ticks=24, seed=seed), sort_keys=False))
                full.append(p)
    (ROOT / "all.txt").write_text("".join(f"{p}\n" for p in full))
    (ROOT / "manifest.json").write_text(json.dumps({"matrix": MATRIX, "modes": MODES, "constants": CONSTANTS,
        "windows": WINDOWS, "seeds": SEEDS, "reuse": "cp12/cp13 from llm-value-allarch-20260523",
        "full_runs": len(full)}, indent=2) + "\n")
    print(json.dumps({"ok": True, "full": len(full), "root": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
