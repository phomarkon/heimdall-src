"""Generate the D6 fill-rate matrix: can the LLM get a higher fill rate at similar profit?

The deterministic baseline proposes the best (highest worst-case-profit) feasible candidate, priced
aggressively -> low fill rate (~0.25), many unfilled bids. A deterministic policy CAN instead chase
fill, but then it sacrifices margin. The LLM question: does a fill-aware LLM find a BETTER point on the
fill/margin frontier — higher fill at similar profit — than either deterministic extreme?

Fair by construction: all three arms select from the IDENTICAL simulator-accepted candidate menu
(preprobe full) using only menu fields (clear_probability_proxy, worst_case_profit) — no extra context
data for the LLM, so any LLM edge is reasoning over the same numbers (honours the data-parity rule).

Arms (same society/seed/window, only the chooser differs):
  * detbest     chooser=deterministic_best_accepted        (margin extreme -> low fill; the baseline)
  * dethighfill chooser=deterministic_high_fill_accepted   (fill extreme  -> high fill, low margin)
  * llmfill     chooser=llm_fill_selector                  (LLM balances fill vs margin)

Positive = llmfill dominates the det frontier: fill rate >> detbest at total profit >= dethighfill.

Usage: PYTHONPATH=. uv run python tools/experiments/generate_d6_fill_20260524.py
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path("ai-society/configs/d6-fill-20260524")
OUT_DIR = "ai-society/runs/d6-fill-20260524"
WINDOWS = {"apr02-0530": "2026-04-02T05:30:00Z", "apr09-1830": "2026-04-09T18:30:00Z"}
SEEDS = [42, 13, 137, 1729, 31415]
# (arm, chooser_mode, llm_enabled)
ARMS = [("detbest", "deterministic_best_accepted", False),
        ("dethighfill", "deterministic_high_fill_accepted", False),
        ("llmfill", "llm_fill_selector", True)]


def _cfg(*, run_id: str, seed: int, ts: str, ticks: int, chooser: str, llm_enabled: bool) -> str:
    return f"""run_id: {run_id}
seed: {seed}
forecaster_seed: {seed}
ticks: {ticks}
start_timestamp: '{ts}'
chooser_mode: {chooser}
preprobe_mode: full
final_bid_guard: simulator_exact_match
ablation_strategy: cp12_llm_suggest_plus_code_ladder
output_dir: {OUT_DIR}
zone: DK1
agent_count: 6
forecaster_backend: f8
forecaster_routing_mode: persona
verifier_mode: simulator
verifier_tau_eur: 0.0
market_context: real
tool_mode: openai_tools
objective: bid_seeking
safety_toolset: full
persona_profile: all_archetypes_v1
scenario_id: p2h_dk1_pypsa
tool_policy: asset_simulator_v1
asset_simulator_mode: scenario_envelope
asset_proxy_style: market
candidate_sizing_mode: large
candidate_sizing_cap_fraction: 1.0
candidate_sizing_min_mwh: 0.25
candidate_sizing_max_candidates: 8
max_tool_rounds: 6
simulator_max_concurrency: 12
data_start: '2026-04-01T00:00:00Z'
data_end: '2026-05-01T00:00:00Z'
context_dataset_dir: data/cache/real_context/april_2026
data_cache_dir: data/cache/real_context/april_2026/source_cache
default_lookback_hours: 24
cache_refresh: false
memory_enabled: false
reviewer_mode: code_only
llm:
  enabled: {str(llm_enabled).lower()}
  base_urls:
  - http://127.0.0.1:8000/v1
  - http://127.0.0.1:8001/v1
  - http://127.0.0.1:8002/v1
  - http://127.0.0.1:8003/v1
  api_key: heimdall-local
  model: Qwen/Qwen3-32B
  temperature: 0.2
  max_tokens: 1024
  timeout_seconds: 180
  max_concurrency: 24
  per_endpoint_max_concurrency: 6
"""


def main() -> None:
    smoke = CONFIG_DIR / "smoke"
    full = CONFIG_DIR / "full"
    smoke.mkdir(parents=True, exist_ok=True)
    full.mkdir(parents=True, exist_ok=True)
    smoke_list, full_list = [], []
    for arm, chooser, le in ARMS:
        rid = f"d6-{arm}-apr02-0530-seed42-2-q32"
        (smoke / f"{rid}.yaml").write_text(_cfg(run_id=rid, seed=42, ts=WINDOWS["apr02-0530"], ticks=2, chooser=chooser, llm_enabled=le))
        smoke_list.append(f"{smoke}/{rid}.yaml")
    for seed in SEEDS:
        for wname, ts in WINDOWS.items():
            for arm, chooser, le in ARMS:
                rid = f"d6-{arm}-{wname}-seed{seed}-24-q32"
                (full / f"{rid}.yaml").write_text(_cfg(run_id=rid, seed=seed, ts=ts, ticks=24, chooser=chooser, llm_enabled=le))
                full_list.append(f"{full}/{rid}.yaml")
    (CONFIG_DIR / "smoke.txt").write_text("\n".join(smoke_list) + "\n")
    (CONFIG_DIR / "full.txt").write_text("\n".join(full_list) + "\n")
    print(f"wrote {len(smoke_list)} smoke + {len(full_list)} full configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
