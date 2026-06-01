"""Fail-fast Exp 2: does a participation-oriented memory unlock capture? GPU4/:8003 only.

The existing memory loop only extracts conservative "watch/guardrail" lessons -> reinforces the
under-participation bottleneck. This tests whether seeding a generic participation-oriented memory
(no leaked window answers) shifts the LLM toward larger feasible bids and fewer needless watches,
raising capture WITHOUT eroding the verifier floor. Controls isolate the trivial value:
  * mem-off           : memory_enabled=false (no in-context lessons)
  * mem-conservative  : the existing timid guardrail bank (the trap-reinforcing control)
  * mem-participation : the curated participation bank (the lever)
Same window/seed/society/chooser as ff1 so it's directly comparable; det baseline = ff1's det arm.

Usage: PYTHONPATH=. uv run python tools/experiments/ff2_memory_20260525.py
"""

from __future__ import annotations

from pathlib import Path

EXPERIMENT = "ff2-memory"
WINDOW = ("apr02-0530", "2026-04-02T05:30:00Z")
SEED, TICKS, PROFILE, AGENTS = 42, 24, "all_archetypes_v1", 6
CONFIG_DIR = Path(f"ai-society/configs/{EXPERIMENT}")
OUT_DIR = f"ai-society/runs/{EXPERIMENT}"
ENDPOINT = "http://127.0.0.1:8003/v1"
CONSERVATIVE_BANK = "ai-society/runs/action-core-matrix/memory-v2-bank.jsonl"
PARTICIPATION_BANK = "ai-society/configs/ff2-memory/participation_bank.jsonl"

# arm -> (memory_enabled, memory_bank_path or "")
ARMS = {
    "mem-off": (False, ""),
    "mem-conservative": (True, CONSERVATIVE_BANK),
    "mem-participation": (True, PARTICIPATION_BANK),
}


def _cfg(*, run_id, mem_on, bank, ticks):
    mem_lines = f"""memory_enabled: {str(mem_on).lower()}"""
    if mem_on:
        mem_lines += f"""
memory_bank_path: {bank}
memory_max_items_per_agent: 5
memory_max_prompt_chars: 2400"""
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
ablation_strategy: cp12_llm_suggest_plus_code_ladder
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
{mem_lines}
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
  max_concurrency: 6
  per_endpoint_max_concurrency: 6
"""


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    full = []
    for arm, (mem_on, bank) in ARMS.items():
        rid = f"{EXPERIMENT}-{arm}-{WINDOW[0]}-seed{SEED}-{TICKS}-q32"
        (CONFIG_DIR / f"{rid}.yaml").write_text(_cfg(run_id=rid, mem_on=mem_on, bank=bank, ticks=TICKS))
        full.append(str(CONFIG_DIR / f"{rid}.yaml"))
    (CONFIG_DIR / "full.txt").write_text("\n".join(full) + "\n")
    print(f"wrote {len(full)} configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
