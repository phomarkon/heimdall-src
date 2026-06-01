"""Count-controlled heterogeneity test (2026-05-25) — runs on GPU4/:8003 ONLY.

De-confounds the §4.9(b) "guarded/det ratio rises with society" lead: across s06/s12/mixed20 the agent
COUNT and composition moved together, so the ratio gradient could be a count artifact, not heterogeneity.
This holds N=12 AND the archetype mix constant, varying ONLY persona/aggressiveness heterogeneity:
  * all_archetypes_double_homo  — LOW het: 2 IDENTICAL copies of each of the 6 archetypes
  * all_archetypes_double_v1 (B)— HIGH het: 2 CONTRASTING copies (averse vs seeking, diff forecaster/size)
If guarded/det(HIGH) > guarded/det(LOW) at fixed N=12, heterogeneity narrows the gap independent of count.

Matched det/guarded x 2 societies x 3 windows x 3 seeds = 36 runs. Pinned to ONE endpoint (:8003 / GPU4)
so it does NOT contend with the 3-GPU generalized-headlines matrix on 8000-8002. Cached Qwen3-32B (quota-safe).

Usage: PYTHONPATH=. uv run python tools/experiments/generate_count_control_20260525.py
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path("ai-society/configs/count-control-20260525")
OUT_DIR = "ai-society/runs/count-control-20260525"
CONTEXT_DIR = "data/cache/real_context/april_2026"
ENDPOINT = "http://127.0.0.1:8003/v1"  # GPU4 only — isolated from the big matrix

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}
SOCIETIES = {  # both N=12, same archetype mix; differ only in persona heterogeneity
    "homo12": "all_archetypes_double_homo",
    "het12": "all_archetypes_double_v1",
}
SEEDS = (42, 13, 137)
VARIANTS = {
    "deterministic": ("deterministic_best_accepted", "schema_only_shadow", False, 0.0, 512),
    "guarded": ("llm", "simulator_exact_match", True, 0.2, 640),
}


def _cfg(*, run_id, profile, variant, ts, seed, ticks):
    chooser, guard, llm_on, temp, maxtok = VARIANTS[variant]
    return f"""run_id: {run_id}
seed: {seed}
forecaster_seed: {seed}
zone: DK1
agent_count: 12
ticks: {ticks}
start_timestamp: '{ts}'
forecaster_backend: f8
forecaster_routing_mode: persona
chooser_mode: {chooser}
verifier_mode: simulator
verifier_tau_eur: 0.0
market_context: real
tool_mode: openai_tools
preprobe_mode: full
objective: bid_seeking
final_bid_guard: {guard}
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
simulator_max_concurrency: 8
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
  enabled: {str(llm_on).lower()}
  base_urls:
  - {ENDPOINT}
  api_key: heimdall-local
  model: Qwen/Qwen3-32B
  temperature: {temp}
  max_tokens: {maxtok}
  timeout_seconds: 180
  max_concurrency: 6
  per_endpoint_max_concurrency: 6
"""


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    full, smoke = [], []
    for seed in SEEDS:
        for soc_slug, profile in SOCIETIES.items():
            for variant in VARIANTS:
                for wname, ts in WINDOWS.items():
                    rid = f"cc-{soc_slug}-{variant}-{wname}-seed{seed}-24-q32"
                    (CONFIG_DIR / f"{rid}.yaml").write_text(_cfg(run_id=rid, profile=profile, variant=variant, ts=ts, seed=seed, ticks=24))
                    full.append(str(CONFIG_DIR / f"{rid}.yaml"))
    for soc_slug, profile in SOCIETIES.items():
        rid = f"cc-{soc_slug}-guarded-apr02-0530-seed42-2-q32"
        (CONFIG_DIR / f"{rid}.yaml").write_text(_cfg(run_id=rid, profile=profile, variant="guarded", ts=WINDOWS["apr02-0530"], seed=42, ticks=2))
        smoke.append(str(CONFIG_DIR / f"{rid}.yaml"))
    (CONFIG_DIR / "full.txt").write_text("\n".join(full) + "\n")
    (CONFIG_DIR / "smoke.txt").write_text("\n".join(smoke) + "\n")
    print(f"wrote {len(smoke)} smoke + {len(full)} full configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
