"""Cross-month ungrounded verifier arm (2026-05-25): the missing shadow-contextonly leg.

The generalized-headlines batch shipped det/guarded/shadow-toolvisible (all grounded) on March, so it
only shows "verifier ~ redundant when grounded". The signature paradox — "verifier load-bearing when
the proposer is UNGROUNDED (high sub-floor)" — was April-only (verifier-ablation-20260524). This adds
the ungrounded arm on the same 2 March windows × 3 seeds × {s06,s12}, directly comparable to the
existing ghv March runs, to make the full paradox cross-month.

shadow-contextonly = chooser=llm, final_bid_guard=schema_only_shadow (verifier OFF, shadow-scored),
safety_toolset=context_only (no simulator tools = ungrounded). Same ablation_strategy / profiles /
endpoints as the generalized-headlines verifier configs.

Usage: PYTHONPATH=. uv run python tools/experiments/generate_verifier_contextonly_march_20260525.py
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path("ai-society/configs/generalized-headlines-20260525")
OUT_DIR = "ai-society/runs/generalized-headlines-20260525"
MODEL = "Qwen/Qwen3-32B"
ENDPOINTS = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1"]
MARCH_CTX = ("2026_03", "2026-03-01T00:00:00Z", "2026-04-01T00:00:00Z")
MARCH_WINDOWS = {"mar11-0000": "2026-03-11T00:00:00Z", "mar24-1915": "2026-03-24T19:15:00Z"}
CDL_SOC = {"s06-actioncore": ("action_core_8", 6), "s12-balanced": ("balanced_intelligence", 12)}
SEEDS = [42, 13, 137]


def _endpoints_yaml() -> str:
    return "\n".join(f"  - {u}" for u in ENDPOINTS)


def _cfg(*, run_id, profile, agent_count, ts, seed, ctx):
    cname, dstart, dend = ctx
    return f"""run_id: {run_id}
seed: {seed}
forecaster_seed: {seed}
zone: DK1
agent_count: {agent_count}
ticks: 24
start_timestamp: '{ts}'
forecaster_backend: f8
forecaster_routing_mode: persona
chooser_mode: llm
verifier_mode: simulator
verifier_tau_eur: 0.0
market_context: real
tool_mode: openai_tools
preprobe_mode: full
objective: bid_seeking
final_bid_guard: schema_only_shadow
safety_toolset: context_only
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
data_start: '{dstart}'
data_end: '{dend}'
context_dataset_dir: data/cache/real_context/{cname}
data_cache_dir: data/cache/real_context/{cname}/source_cache
default_lookback_hours: 24
cache_refresh: false
output_dir: {OUT_DIR}
memory_enabled: false
reviewer_mode: code_only
llm:
  enabled: true
  base_urls:
{_endpoints_yaml()}
  api_key: heimdall-local
  model: {MODEL}
  temperature: 0.2
  max_tokens: 640
  timeout_seconds: 180
  max_concurrency: 18
  per_endpoint_max_concurrency: 6
"""


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for slug, (prof, n) in CDL_SOC.items():
        for seed in SEEDS:
            for wname, ts in MARCH_WINDOWS.items():
                rid = f"ghv-{slug}-shadow-contextonly-{wname}-seed{seed}-24-q32"
                (CONFIG_DIR / f"{rid}.yaml").write_text(_cfg(run_id=rid, profile=prof, agent_count=n, ts=ts, seed=seed, ctx=MARCH_CTX))
                out.append(str(CONFIG_DIR / f"{rid}.yaml"))
    # add the 1 failed sizing rerun
    failed = CONFIG_DIR / "ghs-s20-mixed-large-apr13-0015-seed13-24-q32.yaml"
    if failed.exists():
        out.append(str(failed))
    (CONFIG_DIR / "contextonly_march_plus_rerun.txt").write_text("\n".join(out) + "\n")
    print(f"wrote {len(out)} configs (12 contextonly-March + failed-rerun) -> contextonly_march_plus_rerun.txt")


if __name__ == "__main__":
    main()
