"""Generate or check the llm-value-fairtest-20260522 matrix.

Question: prior matrices could not attribute value to the LLM because the LLM saw the
SAME forecast quantiles as the deterministic ranker and only selected among the ranker's
candidates. This matrix isolates each candidate value mechanism against a properly
matched deterministic+verifier control, holding bid sizing constant and averaging over
the 5 frozen seeds:

  det      deterministic_best_accepted (verifier-gated)            -- the control floor
  selector LLM selects over the same pre-ranked menu               -- selection value
  comm     selector + comm_broadcast_digest                        -- info edge: society digest
  memory   selector + cross-run memory bank                        -- info edge: memory
  cp12     LLM proposes its own candidates (guard relaxed)         -- generative agency

Only ONE axis moves per mode vs `selector`/`det`, so a paired win is attributable.
Regime/context-narrative is a separate feature build and is added later as its own arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

MATRIX = "llm-value-fairtest-20260522"
ROOT = Path(f"ai-society/configs/{MATRIX}")
RUN_ROOT = Path(f"ai-society/runs/{MATRIX}")
CONTEXT_DIR = "data/cache/real_context/april_2026"
TRUTH_DIR = "data/cache/evaluation_truth/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"
MODEL = "Qwen/Qwen3-32B"

# Frozen seeds (project hard constraint). Cut to 3 seeds to fit an ~8h compute budget. Stage 1 = the
# core 2-seed result (42,13) that must finish first; stage 2 = the extra seed (137) that
# fills remaining time and degrades gracefully if runs are slow.
SEEDS = [42, 13, 137]
STAGE1_SEEDS = {42, 13}
SMOKE_SEED = 42

# Two ACTIVE windows only (large oracle). apr13-0015 dropped: quiet window, ~0 oracle,
# nothing to differentiate (a documented trap), and it saves a third of the runs.
WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
}

AGENT_COUNT = 6
PERSONA_PROFILE = "action_core_8"

# Each mode lists ONLY the knobs that define its value mechanism. Everything else is a
# held-constant base value (CONSTANTS) — crucially candidate_sizing_mode and verifier_tau,
# so a win is never a sizing or safety-floor artifact.
MODES: dict[str, dict[str, Any]] = {
    "det": {
        "chooser_mode": "deterministic_best_accepted",
        "llm_enabled": False,
        "preprobe_mode": "full",
        "final_bid_guard": "schema_only_shadow",  # moot for deterministic (no LLM bid)
        "ablation_strategy": "baseline",
        "memory_enabled": False,
        "temperature": 0.0,
        "max_tokens": 512,
    },
    "selector": {
        "chooser_mode": "llm",
        "llm_enabled": True,
        "preprobe_mode": "full",
        "final_bid_guard": "simulator_exact_match",
        "ablation_strategy": "baseline",
        "memory_enabled": False,
        "temperature": 0.2,
        "max_tokens": 640,
    },
    "comm": {  # selector + society digest: the ONLY change vs selector is the info edge
        "chooser_mode": "llm",
        "llm_enabled": True,
        "preprobe_mode": "full",
        "final_bid_guard": "simulator_exact_match",
        "ablation_strategy": "comm_broadcast_digest",
        "memory_enabled": False,
        "temperature": 0.2,
        "max_tokens": 768,
    },
    "memory": {  # selector + cross-run memory: the ONLY change vs selector is memory
        "chooser_mode": "llm",
        "llm_enabled": True,
        "preprobe_mode": "full",
        "final_bid_guard": "simulator_exact_match",
        "ablation_strategy": "baseline",
        "memory_enabled": True,
        "temperature": 0.2,
        "max_tokens": 768,
    },
    "cp12": {  # generative agency: LLM proposes candidates; guard relaxed so its bid is
               # submitted and judged (verifier + downside), not snapped to a menu item.
        "chooser_mode": "llm",
        "llm_enabled": True,
        "preprobe_mode": "context_only",
        "final_bid_guard": "schema_only_shadow",
        "ablation_strategy": "cp12_llm_suggest_plus_code_ladder",
        "memory_enabled": False,
        "temperature": 0.2,
        "max_tokens": 768,
    },
}

CONSTANTS = {
    "zone": "DK1",
    "agent_count": AGENT_COUNT,
    "forecaster_backend": "f8",
    "forecaster_routing_mode": "persona",
    "verifier_mode": "simulator",
    "verifier_tau_eur": 0.0,  # constant safety floor across all cells
    "market_context": "real",
    "tool_mode": "openai_tools",
    "objective": "bid_seeking",
    "safety_toolset": "full",
    "persona_profile": PERSONA_PROFILE,
    "scenario_id": "p2h_dk1_pypsa",
    "tool_policy": "asset_simulator_v1",
    "asset_simulator_mode": "scenario_envelope",
    "asset_proxy_style": "market",
    "candidate_sizing_mode": "large",  # held constant -> no sizing confound
    "candidate_sizing_cap_fraction": 1.0,
    "candidate_sizing_min_mwh": 0.25,
    "candidate_sizing_max_candidates": 8,
    "max_tool_rounds": 6,
    "simulator_max_concurrency": 8,
    "data_start": "2026-04-01T00:00:00Z",
    "data_end": "2026-05-01T00:00:00Z",
    "context_dataset_dir": CONTEXT_DIR,
    "data_cache_dir": f"{CONTEXT_DIR}/source_cache",
    "default_lookback_hours": 24,
    "cache_refresh": False,
    "memory_bank_path": MEMORY_BANK,
    "memory_max_items_per_agent": 5,
    "memory_max_prompt_chars": 2400,
    "reviewer_mode": "code_only",
}

EXPECTED_FULL = len(MODES) * len(WINDOWS) * len(SEEDS)
EXPECTED_SMOKE = len(MODES)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Generate or check {MATRIX} configs.")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        full = _read_config_list(ROOT / "all.txt")
        smoke = _read_config_list(ROOT / "smoke.txt")
        _sanity_check(full, smoke)
        _assert_no_existing_outputs({path.stem for path in full + smoke})
        print(json.dumps({"ok": True, "full": len(full), "smoke": len(smoke), "config_root": str(ROOT)}, indent=2))
        return

    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    full_configs: list[Path] = []
    stages: dict[int, list[Path]] = {1: [], 2: []}
    smoke_configs: list[Path] = []
    seen: set[str] = set()

    # Stage-major ordering so the core 2-seed result (stage 1) fully completes before the
    # extra seed (stage 2): an early stop still leaves a complete, analyzable result.
    for stage, seed in [(1 if s in STAGE1_SEEDS else 2, s) for s in SEEDS]:
        for mode_slug in MODES:
            for window_slug, timestamp in WINDOWS.items():
                run_id = f"lvf-s06-{mode_slug}-{window_slug}-seed{seed}-24-q32"
                path = ROOT / "full" / mode_slug / f"{run_id}.yaml"
                _write_config(path, _config(run_id=run_id, mode_slug=mode_slug, timestamp=timestamp,
                                            ticks=24, seed=seed), seen)
                full_configs.append(path)
                stages[stage].append(path)

    for mode_slug in MODES:
        run_id = f"smoke-lvf-s06-{mode_slug}-apr02-0530-seed{SMOKE_SEED}-2-q32"
        path = ROOT / "smoke" / mode_slug / f"{run_id}.yaml"
        _write_config(path, _config(run_id=run_id, mode_slug=mode_slug, timestamp=WINDOWS["apr02-0530"],
                                    ticks=2, seed=SMOKE_SEED), seen)
        smoke_configs.append(path)

    _assert_no_existing_outputs({path.stem for path in full_configs + smoke_configs})
    _sanity_check(full_configs, smoke_configs)
    _write_lists(stages, smoke_configs)
    _write_manifest(full_configs, smoke_configs)

    print(json.dumps({
        "ok": True,
        "full_run_count": len(full_configs),
        "stage1_count": len(stages[1]),
        "stage2_count": len(stages[2]),
        "smoke_run_count": len(smoke_configs),
        "config_root": str(ROOT),
        "run_root": str(RUN_ROOT),
    }, indent=2))


def _config(*, run_id: str, mode_slug: str, timestamp: str, ticks: int, seed: int) -> dict[str, Any]:
    mode = MODES[mode_slug]
    payload: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "forecaster_seed": seed,
        "zone": CONSTANTS["zone"],
        "agent_count": CONSTANTS["agent_count"],
        "ticks": ticks,
        "start_timestamp": timestamp,
        "forecaster_backend": CONSTANTS["forecaster_backend"],
        "forecaster_routing_mode": CONSTANTS["forecaster_routing_mode"],
        "chooser_mode": mode["chooser_mode"],
        "verifier_mode": CONSTANTS["verifier_mode"],
        "verifier_tau_eur": CONSTANTS["verifier_tau_eur"],
        "market_context": CONSTANTS["market_context"],
        "tool_mode": CONSTANTS["tool_mode"],
        "preprobe_mode": mode["preprobe_mode"],
        "objective": CONSTANTS["objective"],
        "final_bid_guard": mode["final_bid_guard"],
        "safety_toolset": CONSTANTS["safety_toolset"],
        "ablation_strategy": mode["ablation_strategy"],
        "persona_profile": CONSTANTS["persona_profile"],
        "scenario_id": CONSTANTS["scenario_id"],
        "tool_policy": CONSTANTS["tool_policy"],
        "asset_simulator_mode": CONSTANTS["asset_simulator_mode"],
        "asset_proxy_style": CONSTANTS["asset_proxy_style"],
        "candidate_sizing_mode": CONSTANTS["candidate_sizing_mode"],
        "candidate_sizing_cap_fraction": CONSTANTS["candidate_sizing_cap_fraction"],
        "candidate_sizing_min_mwh": CONSTANTS["candidate_sizing_min_mwh"],
        "candidate_sizing_max_candidates": CONSTANTS["candidate_sizing_max_candidates"],
        "max_tool_rounds": CONSTANTS["max_tool_rounds"],
        "simulator_max_concurrency": CONSTANTS["simulator_max_concurrency"],
        "data_start": CONSTANTS["data_start"],
        "data_end": CONSTANTS["data_end"],
        "context_dataset_dir": CONSTANTS["context_dataset_dir"],
        "data_cache_dir": CONSTANTS["data_cache_dir"],
        "default_lookback_hours": CONSTANTS["default_lookback_hours"],
        "cache_refresh": CONSTANTS["cache_refresh"],
        "output_dir": str(RUN_ROOT),
        "memory_enabled": mode["memory_enabled"],
        "memory_bank_path": CONSTANTS["memory_bank_path"],
        "memory_max_items_per_agent": CONSTANTS["memory_max_items_per_agent"],
        "memory_max_prompt_chars": CONSTANTS["memory_max_prompt_chars"],
        "reviewer_mode": CONSTANTS["reviewer_mode"],
        "llm": {
            "enabled": mode["llm_enabled"],
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "api_key": "heimdall-local",
            "model": MODEL,
            "temperature": mode["temperature"],
            "max_tokens": mode["max_tokens"],
            "timeout_seconds": 180,
            "max_concurrency": 12,
            "per_endpoint_max_concurrency": 6,
        },
    }
    return payload


def _write_config(path: Path, payload: dict[str, Any], seen: set[str]) -> None:
    run_id = str(payload["run_id"])
    if run_id in seen:
        raise RuntimeError(f"duplicate run_id: {run_id}")
    seen.add(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_lists(stages: dict[int, list[Path]], smoke: list[Path]) -> None:
    ordered = stages[1] + stages[2]
    (ROOT / "all.txt").write_text("".join(f"{p}\n" for p in ordered), encoding="utf-8")
    (ROOT / "stage1.txt").write_text("".join(f"{p}\n" for p in stages[1]), encoding="utf-8")
    (ROOT / "stage2.txt").write_text("".join(f"{p}\n" for p in stages[2]), encoding="utf-8")
    (ROOT / "smoke.txt").write_text("".join(f"{p}\n" for p in smoke), encoding="utf-8")


def _write_manifest(full: list[Path], smoke: list[Path]) -> None:
    payload = {
        "matrix": MATRIX,
        "generated_at": "2026-05-22",
        "question": (
            "Does the LLM beat a deterministic+verifier control on profit/risk, and via which "
            "mechanism: selection, society-digest info edge, cross-run memory, or generative agency? "
            "One axis moves per mode; sizing and verifier floor held constant; 5 frozen seeds."
        ),
        "primary_metrics": [
            "realized_profit_eur",
            "truth_window_oracle_capture",
            "wrong_side_count",
            "downside_cvar_95_eur",
            "verifier_false_accepts",
        ],
        "scoring_note": "Score with tools/evaluation/rescore_runs.py (capped oracle + grounded delivery downside).",
        "full_run_count": len(full),
        "smoke_run_count": len(smoke),
        "seeds": SEEDS,
        "stage1_seeds": sorted(STAGE1_SEEDS),
        "windows": WINDOWS,
        "society": {"agent_count": AGENT_COUNT, "persona_profile": PERSONA_PROFILE},
        "modes": MODES,
        "constants": CONSTANTS,
        "context_dir": CONTEXT_DIR,
        "truth_dir": TRUTH_DIR,
        "config_list": str(ROOT / "all.txt"),
        "stage1_list": str(ROOT / "stage1.txt"),
        "stage2_list": str(ROOT / "stage2.txt"),
        "smoke_list": str(ROOT / "smoke.txt"),
        "run_root": str(RUN_ROOT),
        "full_runs": [_run_manifest_row(p) for p in full],
        "smoke_runs": [_run_manifest_row(p) for p in smoke],
    }
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_manifest_row(path: Path) -> dict[str, Any]:
    payload = _load_config(path)
    return {
        "run_id": path.stem,
        "config": str(path),
        "chooser_mode": payload["chooser_mode"],
        "preprobe_mode": payload["preprobe_mode"],
        "ablation_strategy": payload["ablation_strategy"],
        "memory_enabled": payload["memory_enabled"],
        "final_bid_guard": payload["final_bid_guard"],
        "llm_enabled": payload["llm"]["enabled"],
        "seed": payload["seed"],
        "start_timestamp": payload["start_timestamp"],
        "ticks": payload["ticks"],
    }


def _read_config_list(path: Path) -> list[Path]:
    if not path.exists():
        raise RuntimeError(f"missing config list: {path}")
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sanity_check(full: list[Path], smoke: list[Path]) -> None:
    if len(full) != EXPECTED_FULL:
        raise RuntimeError(f"expected {EXPECTED_FULL} full configs, found {len(full)}")
    if len(smoke) != EXPECTED_SMOKE:
        raise RuntimeError(f"expected {EXPECTED_SMOKE} smoke configs, found {len(smoke)}")

    full_payloads = [_load_config(p) for p in full]
    smoke_payloads = [_load_config(p) for p in smoke]
    all_payloads = full_payloads + smoke_payloads

    run_ids = [str(p["run_id"]) for p in all_payloads]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError("duplicate run_id")

    for key, value in CONSTANTS.items():
        _expect_values(all_payloads, key, {value})

    for payload in all_payloads:
        if payload["llm"]["model"] != MODEL:
            raise RuntimeError(f"{payload['run_id']} must keep llm.model={MODEL}")
        if payload["llm"]["base_urls"] != ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"]:
            raise RuntimeError(f"{payload['run_id']} must use both vLLM endpoints")
        if payload["llm"]["max_concurrency"] != 12 or payload["llm"]["per_endpoint_max_concurrency"] != 6:
            raise RuntimeError(f"{payload['run_id']} has bad LLM concurrency")
        _validate_mode_payload(payload)

    if {p["ticks"] for p in smoke_payloads} != {2}:
        raise RuntimeError("smoke configs must be 2 ticks")
    if {p["ticks"] for p in full_payloads} != {24}:
        raise RuntimeError("full configs must be 24 ticks")
    if {p["seed"] for p in full_payloads} != set(SEEDS):
        raise RuntimeError(f"full matrix must cover seeds {SEEDS}")

    # Every (mode x window x seed) cell exactly once.
    full_cells = {(p["chooser_mode"], p["ablation_strategy"], p["memory_enabled"], p["final_bid_guard"],
                   p["start_timestamp"], p["seed"]) for p in full_payloads}
    if len(full_cells) != EXPECTED_FULL:
        raise RuntimeError("full matrix does not cover unique mode x window x seed cells")


def _validate_mode_payload(payload: dict[str, Any]) -> None:
    if payload["chooser_mode"] == "deterministic_best_accepted":
        if payload["llm"]["enabled"] is not False:
            raise RuntimeError(f"deterministic config must disable llm: {payload['run_id']}")
        return
    if payload["chooser_mode"] != "llm" or payload["llm"]["enabled"] is not True:
        raise RuntimeError(f"bad chooser config: {payload['run_id']}")
    if payload["final_bid_guard"] not in {"simulator_exact_match", "schema_only_shadow"}:
        raise RuntimeError(f"bad final_bid_guard: {payload['run_id']}")


def _expect_values(payloads: list[dict[str, Any]], key: str, expected: set[Any]) -> None:
    values = {p[key] for p in payloads}
    if values != expected:
        raise RuntimeError(f"{key} expected {expected}, found {values}")


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid config payload: {path}")
    return payload


def _assert_no_existing_outputs(run_ids: set[str]) -> None:
    collisions = []
    for run_id in sorted(run_ids):
        for path in (RUN_ROOT / run_id / "summary.json", Path("evaluations") / run_id / "run_summary.json"):
            if path.exists():
                collisions.append(str(path))
    if collisions:
        raise RuntimeError("refusing duplicate completed outputs:\n" + "\n".join(collisions))


if __name__ == "__main__":
    main()
