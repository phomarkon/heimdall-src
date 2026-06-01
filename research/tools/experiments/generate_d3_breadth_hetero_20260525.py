"""D3 auditability/faithfulness — window-breadth + heterogeneity extension (2026-05-25).

Extends the primary LLM-value positive (grounded vs ungrounded rationale faithfulness, the
confabulation 100%->0% result) along the two axes it is currently thin on:

  1. WINDOWS. The original D3 ran 2 April windows (apr02-0530, apr09-1830). Here we add 3 more
     well-characterised April regimes (down-heavy apr03-1430, top-opportunity apr13-0700,
     low/moderate apr22-0830) so the result spans sparse -> extreme and up- vs down-heavy.
  2. SOCIETY HETEROGENEITY. The result was only ever measured on the 6-agent all_archetypes_v1.
     Two richer societies test whether the auditability claim STRENGTHENS with heterogeneity:
       * Society B (all_archetypes_double_v1, 12 agents) — two of each archetype with contrasting
         aggressiveness. Hypothesis: more diverse decision contexts widen the gap between an LLM
         rationale and any fixed hand-written template (the template-beats-LLM control was
         circular precisely because the 6-agent driver set was small and enumerable), and give
         the ungrounded arm more entities to confabulate about.
       * Society C (all_archetypes_plus_info_v1, 9 agents) — action agents + 3 info specialists
         (market-mechanics, imbalance-analytics, trading-risk). Hypothesis: peer-sourced analysis
         is rationale content no template enumerates (the open-endedness axis) and may improve
         grounded faithfulness.

Honest framing: this is a TEST, not a guaranteed win. Each arm
ships with its controls (det_rich template + selector transcription are scored downstream by
tools/evaluation/evaluate_rationale_faithfulness.py / evaluate_explanation_quality.py). We report
whatever the data says; if heterogeneity does NOT widen the gap, that is the result.

Three frozen seeds [42, 13, 137] (the user-approved 3-seed bar for exploratory society work;
the 2 original windows already hold 5 seeds for Society A).

Outputs three config lists under ai-society/configs/d3-breadth-hetero-20260525/:
  * smoke.txt    — 2-tick fail-fast, one per society, core window, seed 42 (6 configs)
  * now.txt      — Society A x 3 NEW windows x 3 seeds x 2 arms = 18 runs (fast, launch now)
  * overnight.txt— Societies B+C x 5 windows x 3 seeds x 2 arms = 60 runs (heavier, launch later)

Usage: PYTHONPATH=. uv run python tools/experiments/generate_d3_breadth_hetero_20260525.py
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path("ai-society/configs/d3-breadth-hetero-20260525")
OUT_DIR = "ai-society/runs/d3-breadth-hetero-20260525"
SEEDS = [42, 13, 137]

# All April 2026 windows -> the april_2026 real-context cache (data_start/end below match).
NEW_WINDOWS = {
    "apr03-1430": "2026-04-03T14:30:00Z",  # down-heavy high-activity
    "apr13-0700": "2026-04-13T07:00:00Z",  # top-April opportunity / missed-opp stress
    "apr22-0830": "2026-04-22T08:30:00Z",  # low/moderate, many active ticks
}
ORIG_WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",  # high-profit up-heavy (core)
    "apr09-1830": "2026-04-09T18:30:00Z",  # extreme evening spike
}
ALL_WINDOWS = {**ORIG_WINDOWS, **NEW_WINDOWS}

# society slug -> (persona_profile, agent_count)
SOCIETIES = {
    "a": ("all_archetypes_v1", 6),          # focal + 1 of each (the established society)
    "b": ("all_archetypes_double_v1", 12),  # 2 of each, contrasting aggressiveness
    "c": ("all_archetypes_plus_info_v1", 9),  # action agents + 3 info specialists
}


def _cfg(*, run_id: str, seed: int, ts: str, ticks: int, grounded: bool, profile: str, agent_count: int) -> str:
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
agent_count: {agent_count}
forecaster_backend: f8
forecaster_routing_mode: persona
verifier_mode: simulator
verifier_tau_eur: 0.0
market_context: real
tool_mode: openai_tools
objective: bid_seeking
safety_toolset: full
persona_profile: {profile}
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


ARMS = [("ungrounded", False), ("grounded", True)]


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    smoke, now, overnight = [], [], []

    def emit(bucket: list[str], *, soc: str, arm: str, grounded: bool, wname: str, ts: str, seed: int, ticks: int) -> None:
        profile, agent_count = SOCIETIES[soc]
        rid = f"d3bh-{soc}-{arm}-{wname}-seed{seed}-{ticks}-q32"
        (CONFIG_DIR / f"{rid}.yaml").write_text(
            _cfg(run_id=rid, seed=seed, ts=ts, ticks=ticks, grounded=grounded, profile=profile, agent_count=agent_count)
        )
        bucket.append(str(CONFIG_DIR / f"{rid}.yaml"))

    # smoke: 2 ticks, one per society, core window apr02, seed 42, grounded arm
    for soc in SOCIETIES:
        emit(smoke, soc=soc, arm="grounded", grounded=True, wname="apr02-0530", ts=ORIG_WINDOWS["apr02-0530"], seed=42, ticks=2)
        emit(smoke, soc=soc, arm="ungrounded", grounded=False, wname="apr02-0530", ts=ORIG_WINDOWS["apr02-0530"], seed=42, ticks=2)

    # NOW: Society A x 3 NEW windows x 3 seeds x 2 arms = 18
    for seed in SEEDS:
        for wname, ts in NEW_WINDOWS.items():
            for arm, grounded in ARMS:
                emit(now, soc="a", arm=arm, grounded=grounded, wname=wname, ts=ts, seed=seed, ticks=24)

    # OVERNIGHT: Societies B + C x 5 windows x 3 seeds x 2 arms = 60
    for soc in ("b", "c"):
        for seed in SEEDS:
            for wname, ts in ALL_WINDOWS.items():
                for arm, grounded in ARMS:
                    emit(overnight, soc=soc, arm=arm, grounded=grounded, wname=wname, ts=ts, seed=seed, ticks=24)

    (CONFIG_DIR / "smoke.txt").write_text("\n".join(smoke) + "\n")
    (CONFIG_DIR / "now.txt").write_text("\n".join(now) + "\n")
    (CONFIG_DIR / "overnight.txt").write_text("\n".join(overnight) + "\n")
    print(f"wrote {len(smoke)} smoke + {len(now)} now + {len(overnight)} overnight configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
