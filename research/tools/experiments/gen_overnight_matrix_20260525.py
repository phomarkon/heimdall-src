"""ONE overnight matrix (~10h): broad breadth (6 windows) + two full days, on the 2 best societies.

The 2 best new societies from the bigger-societies results are v1 (6) and double (12); doubleinfo (15)
underperformed (info tier dilutes capture) and already has 7-window coverage, so it is excluded here so
that 6 broad windows + 2 full days fit a single overnight. Same regime as the rest (comm_broadcast_digest,
arms det / LLM / LLM+RAG, seed 42, all 4 endpoints).

  Part A — broad   : {v1, double} x 6 NEW 24-tick windows x 3 arms            = 36 runs
  Part B — full day: {v1, double} x {apr13 (high-vol), apr07 (moderate)} 96-tick x 3 arms = 12 runs
  total 48 runs.  Full days = whole-day comparison for the new societies (vs the existing s06/s12/mixed20).

Usage: PYTHONPATH=. uv run python tools/experiments/gen_overnight_matrix_20260525.py
"""

from __future__ import annotations

from pathlib import Path

EXP = "overnight-matrix-20260525"
CONFIG_DIR = Path(f"ai-society/configs/{EXP}")
OUT_DIR = f"ai-society/runs/{EXP}"
ENDPOINTS = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1",
             "http://127.0.0.1:8002/v1", "http://127.0.0.1:8003/v1"]
CORPUS = "ai-society/rag/corpus.jsonl"
RAG_CACHE = "ai-society/rag/cache"
SEED = 42

APRIL_CTX = ("april_2026", "2026-04-01T00:00:00Z", "2026-05-01T00:00:00Z")
MARCH_CTX = ("2026_03", "2026-03-01T00:00:00Z", "2026-04-01T00:00:00Z")
# 6 NEW 24-tick breadth windows (varied hour-of-day, both months; distinct from existing windows).
BROAD_WINDOWS = {
    "apr05-1030": ("2026-04-05T10:30:00Z", APRIL_CTX),
    "apr11-2100": ("2026-04-11T21:00:00Z", APRIL_CTX),
    "apr21-1300": ("2026-04-21T13:00:00Z", APRIL_CTX),
    "apr26-0730": ("2026-04-26T07:30:00Z", APRIL_CTX),
    "mar06-1500": ("2026-03-06T15:00:00Z", MARCH_CTX),
    "mar29-2000": ("2026-03-29T20:00:00Z", MARCH_CTX),
}
# 2 full days (96 ticks = 24h), chosen from April truth: apr13 = highest settlement p95 (508 EUR),
# apr07 = median-volatility day (184 EUR).
FULLDAY_WINDOWS = {
    "apr13-0000": ("2026-04-13T00:00:00Z", APRIL_CTX),
    "apr07-0000": ("2026-04-07T00:00:00Z", APRIL_CTX),
}
SOCIETIES = {"v1": ("all_archetypes_v1", 6), "double": ("all_archetypes_double_v1", 12)}
ARMS = {  # arm -> (chooser, llm_on, rag_on)
    "det": ("deterministic_best_accepted", False, False),
    "norag": ("llm", True, False),
    "rag": ("llm", True, True),
}


def _endpoints_yaml() -> str:
    return "\n".join(f"  - {u}" for u in ENDPOINTS)


def _cfg(*, run_id, profile, agents, start, ctx, chooser, llm_on, rag_on, ticks):
    cname, dstart, dend = ctx
    rag_block = ""
    if rag_on:
        rag_block = (f"rag:\n  enabled: true\n  corpus_path: {CORPUS}\n  backend: dense\n"
                     f"  embedding_model: BAAI/bge-small-en-v1.5\n  device: cpu\n"
                     f"  cache_dir: {RAG_CACHE}\n  top_k: 4\n  max_doc_chars: 700\n")
    return f"""run_id: {run_id}
seed: {SEED}
forecaster_seed: {SEED}
zone: DK1
agent_count: {agents}
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
ablation_strategy: comm_broadcast_digest
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
    # Part A: broad 24-tick. Part B: full-day 96-tick. One list -> one batch.
    for kind, windows, ticks in [("broad", BROAD_WINDOWS, 24), ("fullday", FULLDAY_WINDOWS, 96)]:
        for soc, (profile, agents) in SOCIETIES.items():
            for win, (start, ctx) in windows.items():
                for arm, (chooser, llm_on, rag_on) in ARMS.items():
                    rid = f"bigsoc-{kind}-{soc}-{arm}-{win}-seed{SEED}-{ticks}-q32"
                    (CONFIG_DIR / f"{rid}.yaml").write_text(
                        _cfg(run_id=rid, profile=profile, agents=agents, start=start, ctx=ctx,
                             chooser=chooser, llm_on=llm_on, rag_on=rag_on, ticks=ticks))
                    full.append(str(CONFIG_DIR / f"{rid}.yaml"))
    # full-day runs last so the cheaper breadth lands first (de-risks the overnight)
    (CONFIG_DIR / "full.txt").write_text("\n".join(full) + "\n")
    n_broad = len(SOCIETIES) * len(BROAD_WINDOWS) * len(ARMS)
    n_day = len(SOCIETIES) * len(FULLDAY_WINDOWS) * len(ARMS)
    print(f"wrote {len(full)} configs ({n_broad} broad 24-tick + {n_day} full-day 96-tick) to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
