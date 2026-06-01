"""Generate the freeform-matrix-20260526 cell × window × seed sweep.

Cells test whether LLMs ever propose their own bids outside the seeded
candidate menu (hybrid) or must do so without a seeded preprobe (freeform),
versus the existing selector-only behaviour. Two RAG variants paired.

6 cells x 3 windows x 2 seeds = 36 configs.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

OUT_DIR = Path(__file__).parent
MATRIX = "freeform-matrix-20260526"

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}
SEEDS = [42, 13]

# (cell_name, chooser_mode, preprobe_mode, final_bid_guard, rag_enabled, llm_enabled)
CELLS = [
    ("det",        "deterministic_best_accepted", "full", "simulator_exact_match", False, False),
    ("selector",   "llm",                         "full", "simulator_exact_match", False, True),
    ("hybrid",     "llm",                         "full", "schema_only_shadow",    False, True),
    ("freeform",   "llm",                         "none", "schema_only_shadow",    False, True),
    ("hybrid-rag", "llm",                         "full", "schema_only_shadow",    True,  True),
    ("freeform-rag","llm",                        "none", "schema_only_shadow",    True,  True),
]

LLM_BLOCK = textwrap.dedent(
    """\
    llm:
      enabled: {enabled}
      base_urls:
      - http://127.0.0.1:8000/v1
      - http://127.0.0.1:8001/v1
      - http://127.0.0.1:8002/v1
      - http://127.0.0.1:8003/v1
      api_key: heimdall-local
      model: Qwen/Qwen3-32B
      temperature: 0.2
      max_tokens: 768
      timeout_seconds: 180
      max_concurrency: 24
      per_endpoint_max_concurrency: 6
    """
).rstrip()

RAG_BLOCK = textwrap.dedent(
    """\
    rag:
      enabled: true
      corpus_path: ai-society/rag/corpus.jsonl
      backend: dense
      embedding_model: BAAI/bge-small-en-v1.5
      device: cpu
      cache_dir: ai-society/rag/cache
      top_k: 4
      max_doc_chars: 700
    """
).rstrip()


def emit(cell: str, chooser: str, preprobe: str, guard: str, rag: bool, llm_on: bool,
         window: str, ts: str, seed: int) -> str:
    run_id = f"ff-matrix-{cell}-{window}-seed{seed}-24-q32"
    parts = [
        f"run_id: {run_id}",
        f"seed: {seed}",
        f"forecaster_seed: {seed}",
        "zone: DK1",
        "agent_count: 6",
        "ticks: 24",
        f"start_timestamp: '{ts}'",
        "forecaster_backend: f8",
        "forecaster_routing_mode: persona",
        f"chooser_mode: {chooser}",
        "verifier_mode: simulator",
        "verifier_tau_eur: 0.0",
        "market_context: real",
        "tool_mode: openai_tools",
        f"preprobe_mode: {preprobe}",
        "objective: bid_seeking",
        f"final_bid_guard: {guard}",
        "safety_toolset: full",
        "ablation_strategy: cp12_llm_suggest_plus_code_ladder",
        "persona_profile: all_archetypes_v1",
        "scenario_id: p2h_dk1_pypsa",
        "tool_policy: asset_simulator_v1",
        "asset_simulator_mode: scenario_envelope",
        "asset_proxy_style: market",
        "candidate_sizing_mode: medium",
        "candidate_sizing_cap_fraction: 1.0",
        "candidate_sizing_min_mwh: 0.25",
        "candidate_sizing_max_candidates: 8",
        "max_tool_rounds: 6",
        "simulator_max_concurrency: 8",
        "data_start: '2026-04-01T00:00:00Z'",
        "data_end: '2026-05-01T00:00:00Z'",
        "context_dataset_dir: data/cache/real_context/april_2026",
        "data_cache_dir: data/cache/real_context/april_2026/source_cache",
        "default_lookback_hours: 24",
        "cache_refresh: false",
        f"output_dir: ai-society/runs/{MATRIX}",
        "memory_enabled: false",
        "reviewer_mode: code_only",
    ]
    if rag:
        parts.append(RAG_BLOCK)
    parts.append(LLM_BLOCK.format(enabled=str(llm_on).lower()))
    return "\n".join(parts) + "\n"


def main() -> None:
    config_list: list[str] = []
    for window, ts in WINDOWS.items():
        for seed in SEEDS:
            for cell, chooser, preprobe, guard, rag, llm_on in CELLS:
                run_id = f"ff-matrix-{cell}-{window}-seed{seed}-24-q32"
                path = OUT_DIR / f"{run_id}.yaml"
                path.write_text(emit(cell, chooser, preprobe, guard, rag, llm_on,
                                     window, ts, seed), encoding="utf-8")
                config_list.append(str(path.resolve().relative_to(Path("/home/ucloud/heimdall"))))
    (OUT_DIR / "config-list.txt").write_text("\n".join(config_list) + "\n", encoding="utf-8")
    print(f"wrote {len(config_list)} configs + config-list.txt to {OUT_DIR}")


if __name__ == "__main__":
    main()
