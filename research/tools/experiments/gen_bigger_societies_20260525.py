"""Bigger heterogeneous societies (2026-05-25): within-archetype doubling + info tier.

Two societies, same 7 windows / 1 seed as the rest of the matrix, to see how a denser
heterogeneous population behaves and whether an information-specialist tier changes it:

  * double      = all_archetypes_double_v1            (12: each action archetype x2 with
                                                       contrasting risk / forecaster / size)
  * doubleinfo  = all_archetypes_double_plus_info_v1  (15: the 12 above + 3 info specialists
                                                       — market-mechanics, imbalance, trading-risk)

Ablation = comm_broadcast_digest for ALL arms so info agents actually broadcast analysis the
action agents can cite (with no communication channel the info tier would be inert), and so
double-vs-doubleinfo cleanly isolates "info agents present or not".

Arms (det = clean control; LLM = guarded chooser; RAG = + leak-safe retrieval):
  det        : chooser=deterministic_best_accepted, verifier ON, llm off, rag off
  norag      : chooser=llm,                          verifier ON, llm on,  rag off
  rag        : chooser=llm,                          verifier ON, llm on,  rag ON

Seed 42 only. All 4 endpoints (8000-8003) — run after the GPUs are free.

Usage: PYTHONPATH=. uv run python tools/experiments/gen_bigger_societies_20260525.py
"""

from __future__ import annotations

from pathlib import Path

EXP = "bigger-societies-20260525"
CONFIG_DIR = Path(f"ai-society/configs/{EXP}")
OUT_DIR = f"ai-society/runs/{EXP}"
ENDPOINTS = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1",
             "http://127.0.0.1:8002/v1", "http://127.0.0.1:8003/v1"]
CORPUS = "ai-society/rag/corpus.jsonl"
RAG_CACHE = "ai-society/rag/cache"
SEED = 42

APRIL_CTX = ("april_2026", "2026-04-01T00:00:00Z", "2026-05-01T00:00:00Z")
MARCH_CTX = ("2026_03", "2026-03-01T00:00:00Z", "2026-04-01T00:00:00Z")
WINDOWS = {
    "apr02-0530": ("2026-04-02T05:30:00Z", APRIL_CTX),
    "apr09-1830": ("2026-04-09T18:30:00Z", APRIL_CTX),
    "apr13-0015": ("2026-04-13T00:15:00Z", APRIL_CTX),
    "apr17-1900": ("2026-04-17T19:00:00Z", APRIL_CTX),
    "apr28-1900": ("2026-04-28T19:00:00Z", APRIL_CTX),
    "mar11-0000": ("2026-03-11T00:00:00Z", MARCH_CTX),
    "mar24-1915": ("2026-03-24T19:15:00Z", MARCH_CTX),
}
SOCIETIES = {
    "double": ("all_archetypes_double_v1", 12),
    "doubleinfo": ("all_archetypes_double_plus_info_v1", 15),
}
# arm -> (chooser, llm_on, rag_on)
ARMS = {
    "det": ("deterministic_best_accepted", False, False),
    "norag": ("llm", True, False),
    "rag": ("llm", True, True),
}


def _endpoints_yaml() -> str:
    return "\n".join(f"  - {u}" for u in ENDPOINTS)


def _cfg(*, run_id, profile, agents, start, ctx, chooser, llm_on, rag_on):
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
ticks: 24
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
    full, smoke = [], []
    for soc, (profile, agents) in SOCIETIES.items():
        for win, (start, ctx) in WINDOWS.items():
            for arm, (chooser, llm_on, rag_on) in ARMS.items():
                rid = f"bigsoc-{soc}-{arm}-{win}-seed{SEED}-24-q32"
                (CONFIG_DIR / f"{rid}.yaml").write_text(
                    _cfg(run_id=rid, profile=profile, agents=agents, start=start, ctx=ctx,
                         chooser=chooser, llm_on=llm_on, rag_on=rag_on))
                full.append(str(CONFIG_DIR / f"{rid}.yaml"))
    # smoke: 2-tick rag arm on each society (validates the new profile + comm + rag wiring)
    for soc, (profile, agents) in SOCIETIES.items():
        rid = f"bigsoc-{soc}-rag-apr02-0530-seed{SEED}-2-q32"
        (CONFIG_DIR / f"{rid}.yaml").write_text(
            _cfg(run_id=rid, profile=profile, agents=agents,
                 start=WINDOWS["apr02-0530"][0], ctx=WINDOWS["apr02-0530"][1],
                 chooser="llm", llm_on=True, rag_on=True).replace("ticks: 24", "ticks: 2"))
        smoke.append(str(CONFIG_DIR / f"{rid}.yaml"))
    (CONFIG_DIR / "full.txt").write_text("\n".join(full) + "\n")
    (CONFIG_DIR / "smoke.txt").write_text("\n".join(smoke) + "\n")
    print(f"wrote {len(smoke)} smoke + {len(full)} full configs to {CONFIG_DIR}")
    print(f"societies={list(SOCIETIES)} windows={list(WINDOWS)} arms={list(ARMS)} seed={SEED}")


if __name__ == "__main__":
    main()
