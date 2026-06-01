"""Generate the D3 auditability/faithfulness matrix: ungrounded vs grounded LLM rationale.

Value claim under test: a verifier-guarded LLM can produce a FAITHFUL, evidence-grounded,
auditable rationale — a governance property the deterministic optimizer (which emits no rationale)
structurally cannot. The established D3 negative is that the UNGROUNDED LLM confabulates: it asserts
"no outages / no grid constraints" in ~100% of rationales with zero supporting tool calls. This
matrix tests whether GROUNDING the evidence (seed_outage_context: outages + grid + regime seeded)
eliminates the confabulation, measured by tools/evaluation/evaluate_rationale_faithfulness.py.

Two LLM arms, identical except evidence grounding, on the same windows + frozen seeds:
  * ungrounded : cp12 autonomous, seed_outage_context=false  -> confabulates (control = the problem)
  * grounded   : cp12 autonomous, seed_outage_context=true   -> cites real evidence (the fix)

Fail-fast: a 2-tick smoke per arm on the core window before the 24-tick frozen-seed matrix.

Usage: PYTHONPATH=. uv run python tools/experiments/generate_d3_faithfulness_20260524.py
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path("ai-society/configs/d3-faithfulness-20260524")
OUT_DIR = "ai-society/runs/d3-faithfulness-20260524"
WINDOWS = {"apr02-0530": "2026-04-02T05:30:00Z", "apr09-1830": "2026-04-09T18:30:00Z"}
SEEDS = [42, 13, 137, 1729, 31415]  # frozen set; 42 first


def _cfg(*, run_id: str, seed: int, ts: str, ticks: int, grounded: bool) -> str:
    return f"""run_id: {run_id}
seed: {seed}
forecaster_seed: {seed}
ticks: {ticks}
start_timestamp: '{ts}'
chooser_mode: llm
preprobe_mode: context_only
final_bid_guard: schema_only_shadow
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
seed_outage_context: {str(grounded).lower()}
"""


def main() -> None:
    smoke = CONFIG_DIR / "smoke"
    full = CONFIG_DIR / "full"
    smoke.mkdir(parents=True, exist_ok=True)
    full.mkdir(parents=True, exist_ok=True)
    smoke_list, full_list = [], []

    # 2-tick smoke: both arms, core window, seed 42
    for arm, grounded in [("ungrounded", False), ("grounded", True)]:
        rid = f"d3-{arm}-apr02-0530-seed42-2-q32"
        (smoke / f"{rid}.yaml").write_text(_cfg(run_id=rid, seed=42, ts=WINDOWS["apr02-0530"], ticks=2, grounded=grounded))
        smoke_list.append(f"{smoke}/{rid}.yaml")

    # full 24-tick: both arms x both windows x frozen seeds
    for seed in SEEDS:
        for wname, ts in WINDOWS.items():
            for arm, grounded in [("ungrounded", False), ("grounded", True)]:
                rid = f"d3-{arm}-{wname}-seed{seed}-24-q32"
                (full / f"{rid}.yaml").write_text(_cfg(run_id=rid, seed=seed, ts=ts, ticks=24, grounded=grounded))
                full_list.append(f"{full}/{rid}.yaml")

    (CONFIG_DIR / "smoke.txt").write_text("\n".join(smoke_list) + "\n")
    (CONFIG_DIR / "full.txt").write_text("\n".join(full_list) + "\n")
    print(f"wrote {len(smoke_list)} smoke + {len(full_list)} full configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
