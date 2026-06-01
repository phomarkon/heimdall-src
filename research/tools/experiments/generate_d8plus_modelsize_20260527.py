"""D8+ model-scale extension (2026-05-27): broaden the model ladder + add seeds 13/137.

Standing D8 evidence is seed-42 only across q8/q14/q32/q72 (15 foa-* runs + the
oba-*/lmsm-* fillers). Memory line "(D8 3-seed)" overstates the disk. This batch:

  * Adds **Qwen3-1.7B** (q1.7) as a "very small" rung — currently the smallest Qwen3
    that vLLM supports cleanly with tool-calling. No existing data for this size.
  * Adds seeds 13 + 137 for q8 / q14 / q32 / q72 at the thesis-declared 3 windows
    (apr02-0530 / apr09-1830 / apr13-0015), s06-actioncore + s12-balanced.

Same chooser / verifier / forecaster / window / society as foa-* so cells are
strictly comparable. Mirrors `generate_final_option_a_model_scale.py`.

Storage budget: Qwen3-1.7B (~3.5 GB) is the only new HF pull; q14/q32/q72 are
already cached. /work is at 62 GB / 800 GB cap — well clear.

Run plan (66 new runs; 2 q8 cells overlap with lmsm-* and are skipped by the runner):

  q1.7 : 2 societies × 3 windows × {42,13,137}     = 18
  q8   : 2 societies × 3 windows × {13,137}        = 12
  q14  : 2 societies × 3 windows × {13,137}        = 12
  q32  : 2 societies × 3 windows × {13,137}        = 12
  q72  : 2 societies × 3 windows × {13,137}        = 12

Ordered SMALLEST → LARGEST so vLLM model-swaps are minimized (5 swaps total) and
the small model validates infra before the heavy ones load.

Usage:
  PYTHONPATH=. uv run python tools/experiments/generate_d8plus_modelsize_20260527.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path("ai-society/configs/d8plus-modelsize-20260527")
RUN_ROOT = Path("ai-society/runs/d8plus-modelsize-20260527")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}

# Ordered smallest -> largest so the runner amortizes model-swap cost.
MODELS = {
    "q1.7": "Qwen/Qwen3-1.7B",
    "q8": "Qwen/Qwen3-8B",
    "q14": "Qwen/Qwen3-14B",
    "q32": "Qwen/Qwen3-32B",
    "q72": "Qwen/Qwen2.5-72B-Instruct",
}

# Existing seed42 coverage: q8, q14, q32, q72 across both societies × 3 windows.
# q1.7 has no existing data, so it gets the full 3-seed grid.
SEEDS_BY_MODEL = {
    "q1.7": [42, 13, 137],
    "q8": [13, 137],
    "q14": [13, 137],
    "q32": [13, 137],
    "q72": [13, 137],
}

SOCIETIES = {
    "s06-actioncore": {
        "agent_count": 6,
        "profile": "action_core_8",
        "max_concurrency": 8,
        "per_endpoint_max_concurrency": 4,
        "max_tokens": 1000,
        "memory": False,
    },
    "s12-balanced": {
        "agent_count": 12,
        "profile": "balanced_intelligence",
        "max_concurrency": 12,
        "per_endpoint_max_concurrency": 6,
        "max_tokens": 512,
        "memory": True,
    },
}

SOCIETY_ORDER = ["s06-actioncore", "s12-balanced"]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    configs: list[Path] = []
    smoke_paths: list[Path] = []
    seen: set[str] = set()

    # Smoke: 2-tick s06 run for the smallest model only (catches contract drift
    # before downloading the bigger weights).
    smoke_dir = ROOT / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_id = "d8p-smoke-q1.7-s06-actioncore-apr02-0530-seed42-2"
    smoke_path = smoke_dir / f"{smoke_id}.yaml"
    smoke_path.write_text(
        yaml.safe_dump(
            _config(
                run_id=smoke_id,
                society_slug="s06-actioncore",
                model_slug="q1.7",
                window_slug="apr02-0530",
                seed=42,
                ticks=2,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    smoke_paths.append(smoke_path)

    # Full grid, ordered to minimize model swaps: model -> society -> window -> seed.
    for model_slug in MODELS:
        for society_slug in SOCIETY_ORDER:
            for window_slug in WINDOWS:
                for seed in SEEDS_BY_MODEL[model_slug]:
                    run_id = f"d8p-{society_slug}-bcast-{window_slug}-seed{seed}-{model_slug}"
                    if run_id in seen:
                        raise RuntimeError(f"duplicate run_id: {run_id}")
                    seen.add(run_id)
                    path = ROOT / "full" / model_slug / f"{run_id}.yaml"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        yaml.safe_dump(
                            _config(
                                run_id=run_id,
                                society_slug=society_slug,
                                model_slug=model_slug,
                                window_slug=window_slug,
                                seed=seed,
                                ticks=24,
                            ),
                            sort_keys=False,
                        ),
                        encoding="utf-8",
                    )
                    configs.append(path)

    (ROOT / "config-list.txt").write_text(
        "".join(f"{path}\n" for path in configs), encoding="utf-8"
    )
    (ROOT / "smoke-list.txt").write_text(
        "".join(f"{path}\n" for path in smoke_paths), encoding="utf-8"
    )

    manifest = {
        "run_count": len(configs),
        "smoke_count": len(smoke_paths),
        "ticks": 24,
        "forecaster_backend": "f8",
        "ablation_strategy": "comm_broadcast_digest",
        "windows": list(WINDOWS.keys()),
        "societies": SOCIETY_ORDER,
        "models": MODELS,
        "seeds_by_model": SEEDS_BY_MODEL,
        "runs_by_model": {
            m: sum(1 for p in configs if f"-{m}.yaml" in p.name) for m in MODELS
        },
        "overlaps_to_skip_by_existing_summary": [
            # lmsm-* already covers these two q8 cells (seed-42 path), so the
            # runner will see existing summary.json and skip; documented for
            # auditors.
            "lmsm-s06-actioncore-bcast-apr02-0530-seed13-q8 -- existing as lmsm-*",
            "lmsm-s06-actioncore-bcast-apr02-0530-seed137-q8 -- existing as lmsm-*",
        ],
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _write_runner()
    print(
        json.dumps(
            {
                "ok": True,
                "run_count": len(configs),
                "smoke_count": len(smoke_paths),
                "config_list": str(ROOT / "config-list.txt"),
                "smoke_list": str(ROOT / "smoke-list.txt"),
                "runs_by_model": manifest["runs_by_model"],
            },
            indent=2,
        )
    )


def _config(
    *,
    run_id: str,
    society_slug: str,
    model_slug: str,
    window_slug: str,
    seed: int,
    ticks: int,
) -> dict[str, Any]:
    society = SOCIETIES[society_slug]
    config: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "forecaster_seed": seed,
        "zone": "DK1",
        "agent_count": society["agent_count"],
        "ticks": ticks,
        "start_timestamp": WINDOWS[window_slug],
        "forecaster_backend": "f8",
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "objective": "bid_seeking",
        "ablation_strategy": "comm_broadcast_digest",
        "persona_profile": society["profile"],
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "p2h_only_simulator",
        "max_tool_rounds": 6,
        "data_start": "2026-04-01T00:00:00Z",
        "data_end": "2026-05-01T00:00:00Z",
        "context_dataset_dir": CONTEXT_DIR,
        "data_cache_dir": f"{CONTEXT_DIR}/source_cache",
        "default_lookback_hours": 24,
        "cache_refresh": False,
        "output_dir": str(RUN_ROOT),
        "reviewer_mode": "code_only",
        "llm": {
            "enabled": True,
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "api_key": "heimdall-local",
            "model": MODELS[model_slug],
            "temperature": 0.2,
            "max_tokens": society["max_tokens"],
            "timeout_seconds": 180,
            "max_concurrency": society["max_concurrency"],
            "per_endpoint_max_concurrency": society["per_endpoint_max_concurrency"],
        },
    }
    if society["memory"]:
        config.update(
            {
                "simulator_max_concurrency": 8,
                "memory_enabled": True,
                "memory_bank_path": MEMORY_BANK,
                "memory_max_items_per_agent": 5,
                "memory_max_prompt_chars": 2400,
            }
        )
    return config


def _write_runner() -> None:
    script = RUN_ROOT / "run_d8plus_modelsize.sh"
    smoke = RUN_ROOT / "run_d8plus_smoke.sh"
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /home/ucloud/heimdall",
        "export PYTHONPATH='.:ai-society/src'",
        f"mkdir -p {RUN_ROOT}/logs",
        f'log_dir="{RUN_ROOT}/logs/$(date -u +%Y%m%dT%H%M%SZ)"',
        'mkdir -p "$log_dir"',
        'echo "[$(date -Is)] d8plus model-scale matrix start log_dir=$log_dir"',
        "uv run python ai-society/run_long_model_society_matrix.py \\",
        f"  --config-list {ROOT}/config-list.txt \\",
        '  --log-dir "$log_dir" \\',
        "  --continue-on-failure \\",
        '  > "$log_dir/controller.stdout.log" 2>&1',
        'echo "[$(date -Is)] d8plus model-scale matrix complete log_dir=$log_dir"',
    ]
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)

    smoke_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /home/ucloud/heimdall",
        "export PYTHONPATH='.:ai-society/src'",
        f"mkdir -p {RUN_ROOT}/logs",
        f'log_dir="{RUN_ROOT}/logs/smoke-$(date -u +%Y%m%dT%H%M%SZ)"',
        'mkdir -p "$log_dir"',
        'echo "[$(date -Is)] d8plus smoke start log_dir=$log_dir"',
        "uv run python ai-society/run_long_model_society_matrix.py \\",
        f"  --config-list {ROOT}/smoke-list.txt \\",
        '  --log-dir "$log_dir" \\',
        "  --continue-on-failure \\",
        '  > "$log_dir/controller.stdout.log" 2>&1',
        'echo "[$(date -Is)] d8plus smoke complete log_dir=$log_dir"',
    ]
    smoke.write_text("\n".join(smoke_lines) + "\n", encoding="utf-8")
    smoke.chmod(0o755)


if __name__ == "__main__":
    main()
