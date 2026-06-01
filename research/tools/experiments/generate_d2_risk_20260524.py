"""Generate the D2 variable-focal delivery-risk matrix: risk-blind vs risk-aware LLM.

Value claim under test: the deterministic verifier guards PRICE risk but is structurally BLIND to
DELIVERY risk (`worst_case_profit` assumes perfect delivery). On variable-output assets (wind/
renewables/ev) with real availability uncertainty, a delivery-blind policy over-commits and carries a
fat shortfall tail; an LLM *instructed* about the delivery penalty can size down / abstain to cut the
tail. We test whether the risk-aware LLM improves CVaR of delivery-adjusted profit over the blind one.

Two arms, identical except the delivery-risk instruction, same society + frozen seeds:
  * riskblind : cp12_llm_suggest_plus_code_ladder   (no delivery-risk note — the current default)
  * riskaware : cp12_delivery_risk_aware            (told committed-but-undelivered MWh loses at imbalance)

Scored by tools/evaluation/evaluate_delivery_risk.py (variable-only CVaR, grounded availability).
Society all_archetypes_v1 (wind + 2 renewables + ev variable agents); scenario_envelope.

Usage: PYTHONPATH=. uv run python tools/experiments/generate_d2_risk_20260524.py
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path("ai-society/configs/d2-risk-20260524")
OUT_DIR = "ai-society/runs/d2-risk-20260524"
WINDOWS = {"apr02-0530": "2026-04-02T05:30:00Z", "apr09-1830": "2026-04-09T18:30:00Z"}
SEEDS = [42, 13, 137, 1729, 31415]
ARMS = {"riskblind": "cp12_llm_suggest_plus_code_ladder", "riskaware": "cp12_delivery_risk_aware"}


def _cfg(*, run_id: str, seed: int, ts: str, ticks: int, strategy: str) -> str:
    return f"""run_id: {run_id}
seed: {seed}
forecaster_seed: {seed}
ticks: {ticks}
start_timestamp: '{ts}'
chooser_mode: llm
preprobe_mode: context_only
final_bid_guard: schema_only_shadow
ablation_strategy: {strategy}
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
  enabled: true
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
    for arm, strat in ARMS.items():
        rid = f"d2-{arm}-apr02-0530-seed42-2-q32"
        (smoke / f"{rid}.yaml").write_text(_cfg(run_id=rid, seed=42, ts=WINDOWS["apr02-0530"], ticks=2, strategy=strat))
        smoke_list.append(f"{smoke}/{rid}.yaml")
    for seed in SEEDS:
        for wname, ts in WINDOWS.items():
            for arm, strat in ARMS.items():
                rid = f"d2-{arm}-{wname}-seed{seed}-24-q32"
                (full / f"{rid}.yaml").write_text(_cfg(run_id=rid, seed=seed, ts=ts, ticks=24, strategy=strat))
                full_list.append(f"{full}/{rid}.yaml")
    (CONFIG_DIR / "smoke.txt").write_text("\n".join(smoke_list) + "\n")
    (CONFIG_DIR / "full.txt").write_text("\n".join(full_list) + "\n")
    print(f"wrote {len(smoke_list)} smoke + {len(full_list)} full configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
