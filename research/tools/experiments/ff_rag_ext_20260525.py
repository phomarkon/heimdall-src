"""RAG experiment EXTENSION (2026-05-25): more windows + cross-month, all 4 endpoints.

The first ff-rag run covered apr02-0530 + apr09-1830 only. This adds the remaining April windows used
by the generalized-headlines matrix (apr13-0015, apr17-1900, apr28-1900) plus 2 March windows
(mar11-0000, mar24-1915) for cross-month, seed 42, so RAG is comparable to the headline window
coverage. Uses ALL 4 endpoints (8000-8003) now that the 3-GPU matrix is done.

Arms (the only diff between LLM arms is rag.enabled): det / cp12-norag / cp12-rag.

Usage: PYTHONPATH=. uv run python tools/experiments/ff_rag_ext_20260525.py
"""

from __future__ import annotations

from pathlib import Path

EXPERIMENT = "ff-rag-20260525"  # same family/out dir; new run_ids
CONFIG_DIR = Path(f"ai-society/configs/{EXPERIMENT}")
OUT_DIR = f"ai-society/runs/{EXPERIMENT}"
ENDPOINTS = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1",
             "http://127.0.0.1:8002/v1", "http://127.0.0.1:8003/v1"]
CORPUS = "ai-society/rag/corpus.jsonl"
RAG_CACHE = "ai-society/rag/cache"
SEEDS = [42]
TICKS = 24
PROFILE = "all_archetypes_v1"
AGENTS = 6

APRIL_CTX = ("april_2026", "2026-04-01T00:00:00Z", "2026-05-01T00:00:00Z")
MARCH_CTX = ("2026_03", "2026-03-01T00:00:00Z", "2026-04-01T00:00:00Z")
# window -> (timestamp, ctx)
WINDOWS = {
    "apr13-0015": ("2026-04-13T00:15:00Z", APRIL_CTX),
    "apr17-1900": ("2026-04-17T19:00:00Z", APRIL_CTX),
    "apr28-1900": ("2026-04-28T19:00:00Z", APRIL_CTX),
    "mar11-0000": ("2026-03-11T00:00:00Z", MARCH_CTX),
    "mar24-1915": ("2026-03-24T19:15:00Z", MARCH_CTX),
}
# arm -> (chooser, ablation, llm_on, rag_on)
ARMS = {
    "det": ("deterministic_best_accepted", "deterministic_rich", False, False),
    "cp12-norag": ("llm", "cp12_llm_suggest_plus_code_ladder", True, False),
    "cp12-rag": ("llm", "cp12_llm_suggest_plus_code_ladder", True, True),
}


def _endpoints_yaml() -> str:
    return "\n".join(f"  - {u}" for u in ENDPOINTS)


def _cfg(*, run_id, start, ctx, seed, chooser, ablation, llm_on, rag_on):
    cname, dstart, dend = ctx
    rag_block = ""
    if rag_on:
        rag_block = (f"rag:\n  enabled: true\n  corpus_path: {CORPUS}\n  backend: dense\n"
                     f"  embedding_model: BAAI/bge-small-en-v1.5\n  device: cpu\n"
                     f"  cache_dir: {RAG_CACHE}\n  top_k: 4\n  max_doc_chars: 700\n")
    return f"""run_id: {run_id}
seed: {seed}
forecaster_seed: {seed}
zone: DK1
agent_count: {AGENTS}
ticks: {TICKS}
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
data_start: '{dstart}'
data_end: '{dend}'
context_dataset_dir: data/cache/real_context/{cname}
data_cache_dir: data/cache/real_context/{cname}/source_cache
default_lookback_hours: 24
cache_refresh: false
output_dir: {OUT_DIR}
memory_enabled: false
reviewer_mode: code_only
{rag_block}llm:
  enabled: {str(llm_on).lower()}
  base_urls:
{_endpoints_yaml()}
  api_key: heimdall-local
  model: Qwen/Qwen3-32B
  temperature: 0.2
  max_tokens: 768
  timeout_seconds: 180
  max_concurrency: 24
  per_endpoint_max_concurrency: 6
"""


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    full = []
    for win, (start, ctx) in WINDOWS.items():
        for seed in SEEDS:
            for arm, (chooser, ablation, llm_on, rag_on) in ARMS.items():
                rid = f"{EXPERIMENT}-ext-{arm}-{win}-seed{seed}-{TICKS}-q32"
                (CONFIG_DIR / f"{rid}.yaml").write_text(
                    _cfg(run_id=rid, start=start, ctx=ctx, seed=seed, chooser=chooser,
                         ablation=ablation, llm_on=llm_on, rag_on=rag_on))
                full.append(str(CONFIG_DIR / f"{rid}.yaml"))
    (CONFIG_DIR / "ext_full.txt").write_text("\n".join(full) + "\n")
    print(f"wrote {len(full)} ext configs (windows={list(WINDOWS)} seeds={SEEDS} arms={list(ARMS)})")


if __name__ == "__main__":
    main()
