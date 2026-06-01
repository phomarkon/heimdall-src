"""Fail-fast LLM-value exploration harness (2026-05-25) — GPU4/:8003 ONLY.

Small comparable runs to quickly tell whether a lever adds genuine LLM value over the deterministic
control (no faked metrics — every experiment includes the trivial-value control arm). One window,
seed 42, 24 ticks, all_archetypes_v1 (focal P2H), pinned to :8003 so it never touches the 3-GPU matrix.

Edit EXPERIMENT + ARMS, run, score capture, keep/drop. Usage:
  PYTHONPATH=. uv run python tools/experiments/ff_explore_20260525.py
"""

from __future__ import annotations

from pathlib import Path

EXPERIMENT = "ff1-selector-escape"
WINDOW = ("apr02-0530", "2026-04-02T05:30:00Z")  # high-profit up-heavy; strong signal fast
SEED = 42
TICKS = 24
PROFILE = "all_archetypes_v1"
AGENTS = 6
CONFIG_DIR = Path(f"ai-society/configs/{EXPERIMENT}")
OUT_DIR = f"ai-society/runs/{EXPERIMENT}"
ENDPOINT = "http://127.0.0.1:8003/v1"

# arm -> (chooser_mode, ablation_strategy, final_bid_guard, llm_enabled)
ARMS = {
    "det": ("deterministic_best_accepted", "deterministic_rich", "simulator_exact_match", False),
    "cp12-selector": ("llm", "cp12_llm_suggest_plus_code_ladder", "simulator_exact_match", True),
    "cp11-suggest": ("llm", "cp11_llm_suggest_candidates", "simulator_exact_match", True),
    "cp13-probe-refine": ("llm", "cp13_llm_probe_refine_frontier", "simulator_exact_match", True),
}


def _cfg(*, run_id, chooser, ablation, guard, llm_on, ticks):
    return f"""run_id: {run_id}
seed: {SEED}
forecaster_seed: {SEED}
zone: DK1
agent_count: {AGENTS}
ticks: {ticks}
start_timestamp: '{WINDOW[1]}'
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
  enabled: {str(llm_on).lower()}
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
    full, smoke = [], []
    for arm, (chooser, ablation, guard, llm_on) in ARMS.items():
        rid = f"{EXPERIMENT}-{arm}-{WINDOW[0]}-seed{SEED}-{TICKS}-q32"
        (CONFIG_DIR / f"{rid}.yaml").write_text(_cfg(run_id=rid, chooser=chooser, ablation=ablation, guard=guard, llm_on=llm_on, ticks=TICKS))
        full.append(str(CONFIG_DIR / f"{rid}.yaml"))
    # 1 smoke (2-tick) on the LLM-suggest arm to fail-fast the schema
    rid = f"{EXPERIMENT}-cp11-suggest-{WINDOW[0]}-seed{SEED}-2-q32"
    (CONFIG_DIR / f"{rid}.yaml").write_text(_cfg(run_id=rid, chooser="llm", ablation="cp11_llm_suggest_candidates", guard="simulator_exact_match", llm_on=True, ticks=2))
    smoke.append(str(CONFIG_DIR / f"{rid}.yaml"))
    (CONFIG_DIR / "full.txt").write_text("\n".join(full) + "\n")
    (CONFIG_DIR / "smoke.txt").write_text("\n".join(smoke) + "\n")
    print(f"wrote {len(smoke)} smoke + {len(full)} full configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
