"""D8 model-capability probe: can a BIGGER model reason DIRECTION better (the system's weak point)?

Established baseline: side accuracy is ~0.53-0.64 — at/below the trivial always-down majority class
(0.647). Under the verifier guard the LLM cannot override the forecast-aligned side (cp12 diverges from
det on side 0/122 ticks), so model size can't help there. The only place capability could show
directional value is UNVERIFIED free reasoning (objective=unverified_bid_seeking, safety_toolset=
context_only): the LLM picks side+bid from context with no menu/guard. We compare Qwen3-32B vs the much
larger Qwen2.5-72B-Instruct-AWQ on the same windows/seeds.

Positive = the 72B's free directional accuracy beats both the 32B AND the 0.647 majority-class baseline
(i.e. capability unlocks a real directional signal). Honest null = neither beats majority class →
activation direction is near-unpredictable from the available context, and model size is not the lever.

This is a capability PROBE (unguarded), not a deployment mode. Storage: 72B-AWQ is ~39 GB on /work.

Usage: PYTHONPATH=. uv run python tools/experiments/generate_d8_modelsize_20260524.py
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path("ai-society/configs/d8-modelsize-20260524")
OUT_DIR = "ai-society/runs/d8-modelsize-20260524"
WINDOWS = {"apr02-0530": "2026-04-02T05:30:00Z", "apr09-1830": "2026-04-09T18:30:00Z",
           "apr13-0700": "2026-04-13T07:00:00Z"}
SEEDS = [42, 13, 137]
# arm -> (model, ports)
ARMS = {"q32": ("Qwen/Qwen3-32B", ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1",
                                   "http://127.0.0.1:8002/v1", "http://127.0.0.1:8003/v1"]),
        "q72": ("Qwen/Qwen2.5-72B-Instruct-AWQ", ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1", "http://127.0.0.1:8003/v1"])}


def _cfg(*, run_id: str, seed: int, ts: str, ticks: int, model: str, urls: list[str]) -> str:
    base_urls = "\n".join(f"  - {u}" for u in urls)
    return f"""run_id: {run_id}
seed: {seed}
forecaster_seed: {seed}
ticks: {ticks}
start_timestamp: '{ts}'
chooser_mode: llm
preprobe_mode: context_only
final_bid_guard: schema_only_shadow
ablation_strategy: comm_broadcast_digest
objective: unverified_bid_seeking
safety_toolset: context_only
output_dir: {OUT_DIR}
zone: DK1
agent_count: 6
forecaster_backend: f8
forecaster_routing_mode: persona
verifier_mode: simulator
verifier_tau_eur: 0.0
market_context: real
tool_mode: openai_tools
persona_profile: all_archetypes_v1
scenario_id: p2h_dk1_pypsa
tool_policy: asset_simulator_v1
asset_simulator_mode: scenario_envelope
asset_proxy_style: market
candidate_sizing_mode: large
max_tool_rounds: 4
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
{base_urls}
  api_key: heimdall-local
  model: {model}
  temperature: 0.2
  max_tokens: 1024
  timeout_seconds: 180
  max_concurrency: {12 if len(urls) >= 4 else 6}
  per_endpoint_max_concurrency: 6
"""


def main() -> None:
    smoke = CONFIG_DIR / "smoke"
    full = CONFIG_DIR / "full"
    smoke.mkdir(parents=True, exist_ok=True)
    full.mkdir(parents=True, exist_ok=True)
    lists: dict[str, list[str]] = {"q32": [], "q72": [], "smoke": []}
    for arm, (model, urls) in ARMS.items():
        rid = f"d8-{arm}-apr02-0530-seed42-2-q"
        (smoke / f"{rid}.yaml").write_text(_cfg(run_id=rid, seed=42, ts=WINDOWS["apr02-0530"], ticks=2, model=model, urls=urls))
        lists["smoke"].append(f"{smoke}/{rid}.yaml")
        for seed in SEEDS:
            for wname, ts in WINDOWS.items():
                rid = f"d8-{arm}-{wname}-seed{seed}-24-q"
                (full / f"{rid}.yaml").write_text(_cfg(run_id=rid, seed=seed, ts=ts, ticks=24, model=model, urls=urls))
                lists[arm].append(f"{full}/{rid}.yaml")
    for k, v in lists.items():
        (CONFIG_DIR / f"{k}.txt").write_text("\n".join(v) + "\n")
    print(f"wrote q32={len(lists['q32'])} q72={len(lists['q72'])} smoke={len(lists['smoke'])} configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
