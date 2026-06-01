"""Matched chooser matrix — seed-extension of cdl + heterogeneity-ratio on A/B/C (2026-05-25).

Two products, both matched (same society/seed/window, only chooser+verifier toggle), profit-mode
(preprobe full, bid_seeking), cloning the proven cdl schema (chooser-det-llm-20260522), upgraded to
all 4 vLLM endpoints:

  * cdl_seedext  — make the recovered seed-42 cdl matrix robust: societies {s06-actioncore,
    s12-balanced, mixed20} x {deterministic, guarded, shadow-toolvisible} x 5 windows x seeds[13,137].
  * hetero_bc    — the heterogeneity-ratio test: does guarded/det rise toward (or past) 1.0 as the
    focal-P2H society gets richer? Societies {a=all_archetypes_v1(6), b=all_archetypes_double_v1(12),
    c=all_archetypes_plus_info_v1(9)} x {deterministic, guarded} x 5 windows x seeds[42,13,137].
    Reads directly against the recovered cdl gradient (s06 0.41 -> s12 0.65 -> mixed20 0.77).

Honest: the prior is LLM <= det (selector trap, robust); this tests whether heterogeneity narrows or
closes the gap. Report whatever the ratio says.

Usage: PYTHONPATH=. uv run python tools/experiments/generate_matched_hetero_seedext_20260525.py
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path("ai-society/configs/matched-hetero-seedext-20260525")
OUT_DIR = "ai-society/runs/matched-hetero-seedext-20260525"
CONTEXT_DIR = "data/cache/real_context/april_2026"
MODEL = "Qwen/Qwen3-32B"

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
    "apr17-1900": "2026-04-17T19:00:00Z",
    "apr28-1900": "2026-04-28T19:00:00Z",
}
# cdl seed-extension societies (match the recovered matrix exactly)
CDL_SOC = {
    "s06-actioncore": ("action_core_8", 6),
    "s12-balanced": ("balanced_intelligence", 12),
    "mixed20": ("mixed_expert_20_sideaware", 20),
}
# heterogeneity-ratio societies (focal-P2H A/B/C)
BC_SOC = {
    "a": ("all_archetypes_v1", 6),
    "b": ("all_archetypes_double_v1", 12),
    "c": ("all_archetypes_plus_info_v1", 9),
}
VARIANTS = {
    "deterministic": {"chooser_mode": "deterministic_best_accepted", "final_bid_guard": "schema_only_shadow", "llm_enabled": False, "temperature": 0.0, "max_tokens": 512},
    "guarded": {"chooser_mode": "llm", "final_bid_guard": "simulator_exact_match", "llm_enabled": True, "temperature": 0.2, "max_tokens": 640},
    "shadow-toolvisible": {"chooser_mode": "llm", "final_bid_guard": "schema_only_shadow", "llm_enabled": True, "temperature": 0.2, "max_tokens": 640},
}


def _cfg(*, run_id: str, profile: str, agent_count: int, variant: str, ts: str, seed: int, ticks: int) -> str:
    v = VARIANTS[variant]
    return f"""run_id: {run_id}
seed: {seed}
forecaster_seed: {seed}
zone: DK1
agent_count: {agent_count}
ticks: {ticks}
start_timestamp: '{ts}'
forecaster_backend: f8
forecaster_routing_mode: persona
chooser_mode: {v['chooser_mode']}
verifier_mode: simulator
verifier_tau_eur: 0.0
market_context: real
tool_mode: openai_tools
preprobe_mode: full
objective: bid_seeking
final_bid_guard: {v['final_bid_guard']}
safety_toolset: full
ablation_strategy: comm_broadcast_digest_priority_calibration
persona_profile: {profile}
scenario_id: p2h_dk1_pypsa
tool_policy: asset_simulator_v1
asset_simulator_mode: scenario_envelope
asset_proxy_style: market
candidate_sizing_mode: medium
candidate_sizing_cap_fraction: 1.0
candidate_sizing_min_mwh: 0.25
candidate_sizing_max_candidates: 8
max_tool_rounds: 6
simulator_max_concurrency: 12
data_start: '2026-04-01T00:00:00Z'
data_end: '2026-05-01T00:00:00Z'
context_dataset_dir: {CONTEXT_DIR}
data_cache_dir: {CONTEXT_DIR}/source_cache
default_lookback_hours: 24
cache_refresh: false
output_dir: {OUT_DIR}
memory_enabled: false
reviewer_mode: code_only
llm:
  enabled: {str(v['llm_enabled']).lower()}
  base_urls:
  - http://127.0.0.1:8000/v1
  - http://127.0.0.1:8001/v1
  - http://127.0.0.1:8002/v1
  - http://127.0.0.1:8003/v1
  api_key: heimdall-local
  model: {MODEL}
  temperature: {v['temperature']}
  max_tokens: {v['max_tokens']}
  timeout_seconds: 180
  max_concurrency: 24
  per_endpoint_max_concurrency: 6
"""


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    smoke, cdl_seedext, hetero_bc = [], [], []

    def emit(bucket, *, soc_slug, profile, agent_count, variant, wname, ts, seed, ticks):
        rid = f"mhs-{soc_slug}-{variant}-{wname}-seed{seed}-{ticks}-q32"
        (CONFIG_DIR / f"{rid}.yaml").write_text(_cfg(run_id=rid, profile=profile, agent_count=agent_count, variant=variant, ts=ts, seed=seed, ticks=ticks))
        bucket.append(str(CONFIG_DIR / f"{rid}.yaml"))

    # smoke: 2-tick, one society per group, det+guarded, seed 42, core window
    for slug, (prof, n) in [("s06-actioncore", CDL_SOC["s06-actioncore"]), ("b", BC_SOC["b"]), ("c", BC_SOC["c"])]:
        for variant in ("deterministic", "guarded"):
            emit(smoke, soc_slug=slug, profile=prof, agent_count=n, variant=variant, wname="apr02-0530", ts=WINDOWS["apr02-0530"], seed=42, ticks=2)

    # cdl seed-extension: 3 societies x 3 arms x 5 windows x seeds[13,137]
    for seed in (13, 137):
        for slug, (prof, n) in CDL_SOC.items():
            for variant in VARIANTS:
                for wname, ts in WINDOWS.items():
                    emit(cdl_seedext, soc_slug=slug, profile=prof, agent_count=n, variant=variant, wname=wname, ts=ts, seed=seed, ticks=24)

    # heterogeneity-ratio: A/B/C x {det,guarded} x 5 windows x seeds[42,13,137]
    for seed in (42, 13, 137):
        for slug, (prof, n) in BC_SOC.items():
            for variant in ("deterministic", "guarded"):
                for wname, ts in WINDOWS.items():
                    emit(hetero_bc, soc_slug=slug, profile=prof, agent_count=n, variant=variant, wname=wname, ts=ts, seed=seed, ticks=24)

    (CONFIG_DIR / "smoke.txt").write_text("\n".join(smoke) + "\n")
    (CONFIG_DIR / "cdl_seedext.txt").write_text("\n".join(cdl_seedext) + "\n")
    (CONFIG_DIR / "hetero_bc.txt").write_text("\n".join(hetero_bc) + "\n")
    print(f"wrote {len(smoke)} smoke + {len(cdl_seedext)} cdl_seedext + {len(hetero_bc)} hetero_bc configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
