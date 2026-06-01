"""Sizing seed-extension (2026-05-25): make the large-vs-medium lever robust.

The fco-size result (large ~2x medium profit: s06 48,896 vs 24,932; s20 60,828 vs 32,189) is real and
consistent across both societies and 3 windows, but SEED-42 ONLY. This adds seeds 13 + 137 so the #1
performance lever clears the 3-seed bar. Clones the exact fco-size config
(final-core-overnight-matrix/large_bid_sizing) — only the seed changes — upgraded to all 4 vLLM endpoints.

Societies {s06-actioncore (action_core_8, 6), s20-mixed (mixed_expert_20_sideaware, 20)} x sizes
{medium, large} x windows {apr02-0530, apr09-1830, apr13-0015} x seeds {13, 137} = 24 runs.

Usage: PYTHONPATH=. uv run python tools/experiments/generate_sizing_seedext_20260525.py
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path("ai-society/configs/sizing-seedext-20260525")
OUT_DIR = "ai-society/runs/sizing-seedext-20260525"
CONTEXT_DIR = "data/cache/real_context/april_2026"

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}
SOCIETIES = {
    "s06-actioncore": ("action_core_8", 6),
    "s20-mixed": ("mixed_expert_20_sideaware", 20),
}
SIZES = ("medium", "large")
SEEDS = (13, 137)


def _cfg(*, run_id: str, profile: str, agent_count: int, size: str, ts: str, seed: int, ticks: int) -> str:
    return f"""run_id: {run_id}
seed: {seed}
forecaster_seed: {seed}
zone: DK1
agent_count: {agent_count}
ticks: {ticks}
start_timestamp: '{ts}'
forecaster_backend: f8
chooser_mode: llm
verifier_mode: simulator
verifier_tau_eur: 0.0
market_context: real
tool_mode: openai_tools
objective: bid_seeking
ablation_strategy: comm_broadcast_digest
persona_profile: {profile}
scenario_id: p2h_dk1_pypsa
tool_policy: p2h_only_simulator
max_tool_rounds: 6
candidate_sizing_mode: {size}
candidate_sizing_cap_fraction: 1.0
candidate_sizing_min_mwh: 0.25
candidate_sizing_max_candidates: 8
data_start: '2026-04-01T00:00:00Z'
data_end: '2026-05-01T00:00:00Z'
context_dataset_dir: {CONTEXT_DIR}
data_cache_dir: {CONTEXT_DIR}/source_cache
default_lookback_hours: 24
cache_refresh: false
output_dir: {OUT_DIR}
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
  max_tokens: 1000
  timeout_seconds: 180
  max_concurrency: 24
  per_endpoint_max_concurrency: 6
"""


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    full, smoke = [], []
    for seed in SEEDS:
        for slug, (prof, n) in SOCIETIES.items():
            for size in SIZES:
                for wname, ts in WINDOWS.items():
                    rid = f"szx-{slug}-{size}-{wname}-seed{seed}-24-q32"
                    (CONFIG_DIR / f"{rid}.yaml").write_text(_cfg(run_id=rid, profile=prof, agent_count=n, size=size, ts=ts, seed=seed, ticks=24))
                    full.append(str(CONFIG_DIR / f"{rid}.yaml"))
    # smoke: s06 medium+large, core window, seed13, 2-tick
    for size in SIZES:
        rid = f"szx-s06-actioncore-{size}-apr02-0530-seed13-2-q32"
        (CONFIG_DIR / f"{rid}.yaml").write_text(_cfg(run_id=rid, profile="action_core_8", agent_count=6, size=size, ts=WINDOWS["apr02-0530"], seed=13, ticks=2))
        smoke.append(str(CONFIG_DIR / f"{rid}.yaml"))
    (CONFIG_DIR / "full.txt").write_text("\n".join(full) + "\n")
    (CONFIG_DIR / "smoke.txt").write_text("\n".join(smoke) + "\n")
    print(f"wrote {len(smoke)} smoke + {len(full)} full configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
