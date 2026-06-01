"""Generalized headline batch (2026-05-25): make the 2 headlines robust AND cross-month.

Headlines = (1) the verifier matched matrix (cdl: det/guarded/shadow) and (2) the sizing lever
(large vs medium). Both were seed-42 / April-only. This batch:
  * completes 3 seeds on April (adds seeds 13,137 to the recovered seed-42 results), and
  * adds 2 MARCH windows at 3 seeds -> cross-month GENERALIZATION (the harder month; p99 oracle
    EUR ~3x April). March uses data/cache/real_context/2026_03 (+ evaluation_truth/2026_03, wired
    via the now month-aware run_ablation_batch).

Trimmed for a ~10h run on 3 GPUs (endpoints 8000-8002; GPU4/:8003 reserved for exploration):
  * verifier (cdl): {s06-actioncore, s12-balanced} x {det,guarded,shadow} x
        [April 5win x seeds(13,137)=60] + [March 2win x seeds(42,13,137)=36]  = 96
  * sizing:        {s06-actioncore, s20-mixed} x {medium,large} x
        [April 3win x seeds(13,137)=24] + [March 2win x seeds(42,13,137)=24]  = 48
  total 144 runs. mixed20 (20-agent cdl) dropped to save time; it stays seed-42 from the recovered set.

Usage: PYTHONPATH=. uv run python tools/experiments/generate_generalized_headlines_20260525.py
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path("ai-society/configs/generalized-headlines-20260525")
OUT_DIR = "ai-society/runs/generalized-headlines-20260525"
MODEL = "Qwen/Qwen3-32B"
ENDPOINTS = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1"]

# window -> (timestamp, context_id). context_id selects real_context + (month-aware) evaluation_truth.
APRIL_CTX = ("april_2026", "2026-04-01T00:00:00Z", "2026-05-01T00:00:00Z")
MARCH_CTX = ("2026_03", "2026-03-01T00:00:00Z", "2026-04-01T00:00:00Z")
APRIL_WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z", "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z", "apr17-1900": "2026-04-17T19:00:00Z",
    "apr28-1900": "2026-04-28T19:00:00Z",
}
SIZING_APRIL_WINDOWS = {k: APRIL_WINDOWS[k] for k in ("apr02-0530", "apr09-1830", "apr13-0015")}
MARCH_WINDOWS = {"mar11-0000": "2026-03-11T00:00:00Z", "mar24-1915": "2026-03-24T19:15:00Z"}

CDL_SOC = {"s06-actioncore": ("action_core_8", 6), "s12-balanced": ("balanced_intelligence", 12)}
SIZING_SOC = {"s06-actioncore": ("action_core_8", 6), "s20-mixed": ("mixed_expert_20_sideaware", 20)}
VARIANTS = {
    "deterministic": ("deterministic_best_accepted", "schema_only_shadow", False, 0.0, 512),
    "guarded": ("llm", "simulator_exact_match", True, 0.2, 640),
    "shadow-toolvisible": ("llm", "schema_only_shadow", True, 0.2, 640),
}


def _endpoints_yaml() -> str:
    return "\n".join(f"  - {u}" for u in ENDPOINTS)


def _verifier_cfg(*, run_id, profile, agent_count, variant, ts, seed, ticks, ctx):
    cname, dstart, dend = ctx
    chooser, guard, llm_on, temp, maxtok = VARIANTS[variant]
    return f"""run_id: {run_id}
seed: {seed}
forecaster_seed: {seed}
zone: DK1
agent_count: {agent_count}
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
  enabled: {str(llm_on).lower()}
  base_urls:
{_endpoints_yaml()}
  api_key: heimdall-local
  model: {MODEL}
  temperature: {temp}
  max_tokens: {maxtok}
  timeout_seconds: 180
  max_concurrency: 18
  per_endpoint_max_concurrency: 6
"""


def _sizing_cfg(*, run_id, profile, agent_count, size, ts, seed, ticks, ctx):
    cname, dstart, dend = ctx
    return f"""run_id: {run_id}
seed: {seed}
forecaster_seed: {seed}
zone: DK1
agent_count: {agent_count}
ticks: {ticks}
start_timestamp: '{ts}'
forecaster_backend: f8
chooser_mode: llm
verifier_mode: simulator
verifier_tau_eur: 0.0
market_context: real
tool_mode: openai_tools
objective: bid_seeking
ablation_strategy: comm_broadcast_digest
persona_profile: {profile}
scenario_id: p2h_dk1_pypsa
tool_policy: p2h_only_simulator
max_tool_rounds: 6
candidate_sizing_mode: {size}
candidate_sizing_cap_fraction: 1.0
candidate_sizing_min_mwh: 0.25
candidate_sizing_max_candidates: 8
data_start: '{dstart}'
data_end: '{dend}'
context_dataset_dir: data/cache/real_context/{cname}
data_cache_dir: data/cache/real_context/{cname}/source_cache
default_lookback_hours: 24
cache_refresh: false
output_dir: {OUT_DIR}
reviewer_mode: code_only
llm:
  enabled: true
  base_urls:
{_endpoints_yaml()}
  api_key: heimdall-local
  model: {MODEL}
  temperature: 0.2
  max_tokens: 1000
  timeout_seconds: 180
  max_concurrency: 18
  per_endpoint_max_concurrency: 6
"""


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    full, smoke = [], []

    def w(path: Path, text: str, bucket: list[str]):
        path.write_text(text)
        bucket.append(str(path))

    # verifier (cdl) — April seeds 13,137 + March seeds 42,13,137
    for slug, (prof, n) in CDL_SOC.items():
        for variant in VARIANTS:
            for seed, wins, ctx in [(13, APRIL_WINDOWS, APRIL_CTX), (137, APRIL_WINDOWS, APRIL_CTX),
                                    (42, MARCH_WINDOWS, MARCH_CTX), (13, MARCH_WINDOWS, MARCH_CTX), (137, MARCH_WINDOWS, MARCH_CTX)]:
                for wname, ts in wins.items():
                    rid = f"ghv-{slug}-{variant}-{wname}-seed{seed}-24-q32"
                    w(CONFIG_DIR / f"{rid}.yaml", _verifier_cfg(run_id=rid, profile=prof, agent_count=n, variant=variant, ts=ts, seed=seed, ticks=24, ctx=ctx), full)

    # sizing — April seeds 13,137 + March seeds 42,13,137
    for slug, (prof, n) in SIZING_SOC.items():
        for size in ("medium", "large"):
            for seed, wins, ctx in [(13, SIZING_APRIL_WINDOWS, APRIL_CTX), (137, SIZING_APRIL_WINDOWS, APRIL_CTX),
                                    (42, MARCH_WINDOWS, MARCH_CTX), (13, MARCH_WINDOWS, MARCH_CTX), (137, MARCH_WINDOWS, MARCH_CTX)]:
                for wname, ts in wins.items():
                    rid = f"ghs-{slug}-{size}-{wname}-seed{seed}-24-q32"
                    w(CONFIG_DIR / f"{rid}.yaml", _sizing_cfg(run_id=rid, profile=prof, agent_count=n, size=size, ts=ts, seed=seed, ticks=24, ctx=ctx), full)

    # smoke: 1 April + 1 March, verifier guarded s06 + sizing large s06, 2-tick (validates month-aware path)
    w(CONFIG_DIR / "smoke-ghv-apr.yaml", _verifier_cfg(run_id="smoke-ghv-apr", profile="action_core_8", agent_count=6, variant="guarded", ts=APRIL_WINDOWS["apr02-0530"], seed=42, ticks=2, ctx=APRIL_CTX), smoke)
    w(CONFIG_DIR / "smoke-ghv-mar.yaml", _verifier_cfg(run_id="smoke-ghv-mar", profile="action_core_8", agent_count=6, variant="guarded", ts=MARCH_WINDOWS["mar11-0000"], seed=42, ticks=2, ctx=MARCH_CTX), smoke)
    w(CONFIG_DIR / "smoke-ghs-mar.yaml", _sizing_cfg(run_id="smoke-ghs-mar", profile="action_core_8", agent_count=6, size="large", ts=MARCH_WINDOWS["mar24-1915"], seed=42, ticks=2, ctx=MARCH_CTX), smoke)

    (CONFIG_DIR / "full.txt").write_text("\n".join(full) + "\n")
    (CONFIG_DIR / "smoke.txt").write_text("\n".join(smoke) + "\n")
    print(f"wrote {len(smoke)} smoke + {len(full)} full configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
