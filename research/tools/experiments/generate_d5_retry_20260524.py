"""Generate the D5 autonomous-retry matrix: retry vs no-retry (does self-reconsideration add value?).

Value claim under test: an autonomous agent that can RETRY recovers profitable participation that a
one-shot agent abstains on. comm_retry_council re-prompts an agent that chose watch/abstain when a
simulator-accepted candidate exists, letting it upgrade watch->bid (the only mode with mid-sequence
decision modification). comm_peer_signal is the identical sequential flow WITHOUT the retry step — the
clean control. On the grounded all_archetypes_v1 society (P2H focal) we measure capture (capacity oracle)
+ recovery (watch->bid upgrades) + verifier false-accepts (retry must not buy participation with risk).

Positive = retry raises capture/participation over no-retry WITHOUT raising verifier false-accepts.

Usage: PYTHONPATH=. uv run python tools/experiments/generate_d5_retry_20260524.py
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path("ai-society/configs/d5-retry-20260524")
OUT_DIR = "ai-society/runs/d5-retry-20260524"
WINDOWS = {"apr02-0530": "2026-04-02T05:30:00Z", "apr09-1830": "2026-04-09T18:30:00Z"}
SEEDS = [42, 13, 137, 1729, 31415]
ARMS = {"noretry": "comm_peer_signal", "retry": "comm_retry_council"}


def _cfg(*, run_id: str, seed: int, ts: str, ticks: int, strategy: str) -> str:
    return f"""run_id: {run_id}
seed: {seed}
forecaster_seed: {seed}
ticks: {ticks}
start_timestamp: '{ts}'
chooser_mode: llm
preprobe_mode: full
final_bid_guard: simulator_exact_match
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
        rid = f"d5-{arm}-apr02-0530-seed42-2-q32"
        (smoke / f"{rid}.yaml").write_text(_cfg(run_id=rid, seed=42, ts=WINDOWS["apr02-0530"], ticks=2, strategy=strat))
        smoke_list.append(f"{smoke}/{rid}.yaml")
    for seed in SEEDS:
        for wname, ts in WINDOWS.items():
            for arm, strat in ARMS.items():
                rid = f"d5-{arm}-{wname}-seed{seed}-24-q32"
                (full / f"{rid}.yaml").write_text(_cfg(run_id=rid, seed=seed, ts=ts, ticks=24, strategy=strat))
                full_list.append(f"{full}/{rid}.yaml")
    (CONFIG_DIR / "smoke.txt").write_text("\n".join(smoke_list) + "\n")
    (CONFIG_DIR / "full.txt").write_text("\n".join(full_list) + "\n")
    print(f"wrote {len(smoke_list)} smoke + {len(full_list)} full configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
