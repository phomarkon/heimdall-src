"""Fail-fast Exp 4: does info-specialist -> action coordination add capture? GPU4/:8003 only.

Tests whether routing info-specialist analysis to action agents (comm_info_then_action) beats the same
society under plain broadcast — i.e. whether peer information the deterministic grid lacks helps. Society
all_archetypes_plus_info_v1 (6 action + 3 info specialists). Compared to the ff1 det baseline (22,212 /
0.189). Same window/seed.

Usage: PYTHONPATH=. uv run python tools/experiments/ff4_coordination_20260525.py
"""

from __future__ import annotations

from pathlib import Path

EXPERIMENT = "ff4-coordination"
WINDOW = ("apr02-0530", "2026-04-02T05:30:00Z")
SEED, TICKS, PROFILE, AGENTS = 42, 24, "all_archetypes_plus_info_v1", 9
CONFIG_DIR = Path(f"ai-society/configs/{EXPERIMENT}")
OUT_DIR = f"ai-society/runs/{EXPERIMENT}"
ENDPOINT = "http://127.0.0.1:8003/v1"
ARMS = {  # arm -> ablation_strategy
    "info-then-action": "comm_info_then_action",
    "broadcast-infosoc": "comm_broadcast_digest",
}


def _cfg(*, run_id, ablation, ticks):
    return f"""run_id: {run_id}
seed: {SEED}
forecaster_seed: {SEED}
zone: DK1
agent_count: {AGENTS}
ticks: {ticks}
start_timestamp: '{WINDOW[1]}'
forecaster_backend: f8
forecaster_routing_mode: persona
chooser_mode: llm
verifier_mode: simulator
verifier_tau_eur: 0.0
market_context: real
tool_mode: openai_tools
preprobe_mode: full
objective: bid_seeking
final_bid_guard: simulator_exact_match
safety_toolset: full
ablation_strategy: {ablation}
persona_profile: {PROFILE}
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
context_dataset_dir: data/cache/real_context/april_2026
data_cache_dir: data/cache/real_context/april_2026/source_cache
default_lookback_hours: 24
cache_refresh: false
output_dir: {OUT_DIR}
memory_enabled: false
reviewer_mode: code_only
llm:
  enabled: true
  base_urls:
  - {ENDPOINT}
  api_key: heimdall-local
  model: Qwen/Qwen3-32B
  temperature: 0.2
  max_tokens: 768
  timeout_seconds: 180
  max_concurrency: 9
  per_endpoint_max_concurrency: 9
"""


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    full = []
    for arm, ablation in ARMS.items():
        rid = f"{EXPERIMENT}-{arm}-{WINDOW[0]}-seed{SEED}-{TICKS}-q32"
        (CONFIG_DIR / f"{rid}.yaml").write_text(_cfg(run_id=rid, ablation=ablation, ticks=TICKS))
        full.append(str(CONFIG_DIR / f"{rid}.yaml"))
    (CONFIG_DIR / "full.txt").write_text("\n".join(full) + "\n")
    print(f"wrote {len(full)} configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
