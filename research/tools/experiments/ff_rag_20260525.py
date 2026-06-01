"""RAG LLM-value experiment generator (2026-05-25) — GPU4 / :8003 ONLY.

Tests whether a leak-safe RAG capability (retrieve_knowledge over historical
market stats + prior-run lessons + methodology) improves the focal society over
a no-RAG control and the deterministic control, on the same windows/seeds as the
normal LLM experiments. Three matched arms per (window, seed); the ONLY thing
that differs between the two LLM arms is rag.enabled.

  det        — deterministic_best_accepted / deterministic_rich (capture baseline)
  cp12-norag — LLM selector over the code grid, RAG OFF (selector-trap control)
  cp12-rag   — identical to cp12-norag but RAG ON (retrieve_knowledge offered)

Pinned to http://127.0.0.1:8003/v1 so it never touches the 3-GPU matrix.

Usage:
  PYTHONPATH=. uv run python tools/experiments/ff_rag_20260525.py
then launch the printed config lists with ai-society/run_ablation_batch.py.
"""

from __future__ import annotations

from pathlib import Path

EXPERIMENT = "ff-rag-20260525"
# Same windows the normal LLM matrix uses; start with apr02-0530 (ff standard).
WINDOWS = [
    ("apr02-0530", "2026-04-02T05:30:00Z"),
    ("apr09-1830", "2026-04-09T18:30:00Z"),
]
SEEDS = [42, 13, 137]
TICKS = 24
PROFILE = "all_archetypes_v1"
AGENTS = 6
CONFIG_DIR = Path(f"ai-society/configs/{EXPERIMENT}")
OUT_DIR = f"ai-society/runs/{EXPERIMENT}"
ENDPOINT = "http://127.0.0.1:8003/v1"
CORPUS = "ai-society/rag/corpus.jsonl"
RAG_CACHE = "ai-society/rag/cache"

# arm -> (chooser_mode, ablation_strategy, llm_enabled, rag_enabled)
ARMS = {
    "det": ("deterministic_best_accepted", "deterministic_rich", False, False),
    "cp12-norag": ("llm", "cp12_llm_suggest_plus_code_ladder", True, False),
    "cp12-rag": ("llm", "cp12_llm_suggest_plus_code_ladder", True, True),
}


def _cfg(*, run_id, start, seed, chooser, ablation, llm_on, rag_on, ticks):
    rag_block = ""
    if rag_on:
        rag_block = f"""rag:
  enabled: true
  corpus_path: {CORPUS}
  backend: dense
  embedding_model: BAAI/bge-small-en-v1.5
  device: cpu
  cache_dir: {RAG_CACHE}
  top_k: 4
  max_doc_chars: 700
"""
    return f"""run_id: {run_id}
seed: {seed}
forecaster_seed: {seed}
zone: DK1
agent_count: {AGENTS}
ticks: {ticks}
start_timestamp: '{start}'
forecaster_backend: f8
forecaster_routing_mode: persona
chooser_mode: {chooser}
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
{rag_block}llm:
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
    for win_label, start in WINDOWS:
        for seed in SEEDS:
            for arm, (chooser, ablation, llm_on, rag_on) in ARMS.items():
                rid = f"{EXPERIMENT}-{arm}-{win_label}-seed{seed}-{TICKS}-q32"
                (CONFIG_DIR / f"{rid}.yaml").write_text(
                    _cfg(run_id=rid, start=start, seed=seed, chooser=chooser,
                         ablation=ablation, llm_on=llm_on, rag_on=rag_on, ticks=TICKS)
                )
                full.append(str(CONFIG_DIR / f"{rid}.yaml"))
    # 1 smoke (2-tick) on the RAG arm to fail-fast the tool wiring + leakage
    rid = f"{EXPERIMENT}-cp12-rag-{WINDOWS[0][0]}-seed42-2-q32"
    (CONFIG_DIR / f"{rid}.yaml").write_text(
        _cfg(run_id=rid, start=WINDOWS[0][1], seed=42, chooser="llm",
             ablation="cp12_llm_suggest_plus_code_ladder", llm_on=True, rag_on=True, ticks=2)
    )
    smoke.append(str(CONFIG_DIR / f"{rid}.yaml"))
    (CONFIG_DIR / "full.txt").write_text("\n".join(full) + "\n")
    (CONFIG_DIR / "smoke.txt").write_text("\n".join(smoke) + "\n")
    print(f"wrote {len(smoke)} smoke + {len(full)} full configs to {CONFIG_DIR}")
    print(f"windows={[w[0] for w in WINDOWS]} seeds={SEEDS} arms={list(ARMS)}")


if __name__ == "__main__":
    main()
