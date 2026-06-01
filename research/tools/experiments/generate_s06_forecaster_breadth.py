from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path("ai-society/configs/s06-forecaster-breadth-20260520")
RUN_ROOT = Path("ai-society/runs/s06-forecaster-breadth-20260520")
UPSTREAM_ROOT = Path("ai-society/configs/scenario-envelope-thesis-ablation-20260520")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"
MODEL = "Qwen/Qwen3-32B"

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}
INTENDED_FORECASTERS = (
    "ar1",
    "f0",
    "f1_lgbm",
    "f2_blr",
    "f3_ensemble",
    "f3_lite",
    "f4_mc_dropout",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
)
SMOKE_FORECASTERS = ("f0", "f8", "f10")
EXPECTED_FULL = 36
EXPECTED_SMOKE = 3
DEFAULT_SEED = 42

NEURAL_OR_CHECKPOINT = {
    "f1_lgbm",
    "f2_blr",
    "f3_ensemble",
    "f3_lite",
    "f4_mc_dropout",
    "f7",
    "f8",
    "f11",
}
FOUNDATION = {"f9", "f10"}
SIMPLE = {"ar1", "f0"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or check s06 forecaster breadth configs.")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        full = _read_config_list(ROOT / "all.txt")
        smoke = _read_config_list(ROOT / "smoke.txt")
        manifest = _read_manifest()
        fallback_plan = _validate_forecasters()
        _sanity_check(full, smoke, manifest, fallback_plan)
        _assert_no_existing_outputs({path.stem for path in full + smoke})
        print(json.dumps({"ok": True, "full": len(full), "smoke": len(smoke), "config_root": str(ROOT)}, indent=2))
        return

    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    fallback_plan = _validate_forecasters()
    full_configs: list[Path] = []
    smoke_configs: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []
    smoke_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for window_slug, timestamp in WINDOWS.items():
        for intended in INTENDED_FORECASTERS:
            actual = fallback_plan[intended]["actual_forecaster"]
            run_id = _run_id("s06fb", intended, actual, window_slug, 24)
            path = ROOT / "full" / intended / f"{run_id}.yaml"
            payload = _config(run_id=run_id, timestamp=timestamp, forecaster=actual, ticks=24)
            _write_config(path, payload, seen)
            full_configs.append(path)
            manifest_rows.append(_manifest_row(path, intended, fallback_plan[intended]))

    for intended in SMOKE_FORECASTERS:
        actual = fallback_plan[intended]["actual_forecaster"]
        run_id = _run_id("smoke-s06fb", intended, actual, "apr02-0530", 2)
        path = ROOT / "smoke" / intended / f"{run_id}.yaml"
        payload = _config(run_id=run_id, timestamp=WINDOWS["apr02-0530"], forecaster=actual, ticks=2)
        _write_config(path, payload, seen)
        smoke_configs.append(path)
        smoke_rows.append(_manifest_row(path, intended, fallback_plan[intended]))

    _assert_no_existing_outputs({path.stem for path in full_configs + smoke_configs})
    _sanity_check(full_configs, smoke_configs, {"full_runs": manifest_rows, "smoke_runs": smoke_rows}, fallback_plan)
    _write_lists(full_configs, smoke_configs)
    _write_manifest(manifest_rows, smoke_rows, fallback_plan)
    _write_runbook()
    _write_launcher()

    print(
        json.dumps(
            {
                "ok": True,
                "full_run_count": len(full_configs),
                "smoke_run_count": len(smoke_configs),
                "config_root": str(ROOT),
                "run_root": str(RUN_ROOT),
                "fallbacks": {
                    key: value for key, value in fallback_plan.items() if value["fallback_used"]
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def _validate_forecasters() -> dict[str, dict[str, Any]]:
    from heimdall_forecaster.inference import clear_cache, get_forecaster, list_registered

    registered = set(list_registered())
    plan: dict[str, dict[str, Any]] = {}
    for intended in INTENDED_FORECASTERS:
        if intended not in registered:
            plan[intended] = _fallback_record(intended, f"{intended} is not registered")
            continue
        try:
            clear_cache()
            forecaster = get_forecaster(intended, seed=DEFAULT_SEED)
            _probe_predict(forecaster)
            internal_note = None
            if intended == "f10" and getattr(forecaster, "_fallback", None) is not None:
                internal_note = "f10 instantiated, but Chronos-Bolt optional dependency was unavailable; local F9 TimesFM fallback served predictions."
            plan[intended] = {
                "intended_forecaster": intended,
                "actual_forecaster": intended,
                "fallback_used": False,
                "fallback_reason": None,
                "internal_fallback_note": internal_note,
            }
        except Exception as exc:
            plan[intended] = _fallback_record(intended, f"{type(exc).__name__}: {str(exc).splitlines()[0]}")

    for record in plan.values():
        actual = str(record["actual_forecaster"])
        try:
            clear_cache()
            forecaster = get_forecaster(actual, seed=DEFAULT_SEED)
            _probe_predict(forecaster)
        except Exception as exc:
            raise RuntimeError(
                f"actual fallback backend {actual!r} for intended {record['intended_forecaster']!r} is not loadable: {exc}"
            ) from exc
    return plan


def _probe_predict(forecaster: Any) -> None:
    history = np.linspace(10.0, 25.0, 96, dtype=float)
    forecast = forecaster.predict(history, horizon=4)
    if len(forecast) != 4:
        raise RuntimeError(f"expected 4 forecast steps, got {len(forecast)}")


def _fallback_record(intended: str, reason: str) -> dict[str, Any]:
    actual = _fallback_for(intended)
    return {
        "intended_forecaster": intended,
        "actual_forecaster": actual,
        "fallback_used": True,
        "fallback_reason": reason,
        "internal_fallback_note": None,
    }


def _fallback_for(intended: str) -> str:
    if intended in FOUNDATION:
        return "f9"
    if intended in SIMPLE:
        return "f0"
    if intended in NEURAL_OR_CHECKPOINT:
        return "f8"
    return "f8"


def _run_id(prefix: str, intended: str, actual: str, window: str, ticks: int) -> str:
    forecaster_slug = intended if intended == actual else f"{intended}-as-{actual}"
    return f"{prefix}-{forecaster_slug}-scenario-large-{window}-{ticks}-q32"


def _config(*, run_id: str, timestamp: str, forecaster: str, ticks: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": DEFAULT_SEED,
        "forecaster_seed": DEFAULT_SEED,
        "zone": "DK1",
        "agent_count": 6,
        "ticks": ticks,
        "start_timestamp": timestamp,
        "forecaster_backend": forecaster,
        "forecaster_routing_mode": "run_level",
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "preprobe_mode": "full",
        "objective": "bid_seeking",
        "final_bid_guard": "simulator_exact_match",
        "safety_toolset": "full",
        "ablation_strategy": "comm_broadcast_digest_priority_calibration",
        "persona_profile": "action_core_8",
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "asset_simulator_v1",
        "asset_simulator_mode": "scenario_envelope",
        "asset_proxy_style": "market",
        "candidate_sizing_mode": "large",
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
        "output_dir": str(RUN_ROOT),
        "memory_enabled": False,
        "memory_bank_path": MEMORY_BANK,
        "memory_max_items_per_agent": 5,
        "memory_max_prompt_chars": 2400,
        "reviewer_mode": "code_only",
        "llm": {
            "enabled": True,
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "api_key": "heimdall-local",
            "model": MODEL,
            "temperature": 0.2,
            "max_tokens": 640,
            "timeout_seconds": 180,
            "max_concurrency": 8,
            "per_endpoint_max_concurrency": 4,
        },
    }


def _write_config(path: Path, payload: dict[str, Any], seen: set[str]) -> None:
    run_id = str(payload["run_id"])
    if run_id in seen:
        raise RuntimeError(f"duplicate run_id: {run_id}")
    seen.add(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_lists(full_configs: list[Path], smoke_configs: list[Path]) -> None:
    (ROOT / "all.txt").write_text("".join(f"{path}\n" for path in full_configs), encoding="utf-8")
    (ROOT / "smoke.txt").write_text("".join(f"{path}\n" for path in smoke_configs), encoding="utf-8")


def _write_manifest(
    full_rows: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
    fallback_plan: dict[str, dict[str, Any]],
) -> None:
    payload = {
        "matrix": "s06-forecaster-breadth-20260520",
        "generated_at": "2026-05-20",
        "full_run_count": len(full_rows),
        "smoke_run_count": len(smoke_rows),
        "default_seed": DEFAULT_SEED,
        "windows": WINDOWS,
        "society": "s06-actioncore",
        "agent_count": 6,
        "persona_profile": "action_core_8",
        "intended_forecasters": list(INTENDED_FORECASTERS),
        "excluded_forecasters": {"f12": "not loadable locally: missing stats.pkl"},
        "fallback_policy": {
            "neural_or_checkpoint": "f8",
            "foundation_or_zero_shot": "f9",
            "simple_baseline": "f0",
        },
        "forecaster_validation": fallback_plan,
        "asset_simulator_mode": "scenario_envelope",
        "candidate_sizing_mode": "large",
        "chain_after": {
            "matrix": "scenario-envelope-thesis-ablation-20260520",
            "required_completed": 30,
            "required_failed": 0,
        },
        "config_list": str(ROOT / "all.txt"),
        "smoke_list": str(ROOT / "smoke.txt"),
        "run_root": str(RUN_ROOT),
        "full_runs": full_rows,
        "smoke_runs": smoke_rows,
    }
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_row(path: Path, intended: str, record: dict[str, Any]) -> dict[str, Any]:
    payload = _load_config(path)
    return {
        "run_id": path.stem,
        "config": str(path),
        "intended_forecaster": intended,
        "actual_forecaster": record["actual_forecaster"],
        "fallback_used": record["fallback_used"],
        "fallback_reason": record["fallback_reason"],
        "internal_fallback_note": record["internal_fallback_note"],
        "ticks": payload["ticks"],
        "start_timestamp": payload["start_timestamp"],
        "seed": payload["seed"],
        "asset_simulator_mode": payload["asset_simulator_mode"],
        "candidate_sizing_mode": payload["candidate_sizing_mode"],
        "forecaster_routing_mode": payload["forecaster_routing_mode"],
        "persona_profile": payload["persona_profile"],
    }


def _write_runbook() -> None:
    (ROOT / "RUNBOOK.md").write_text(RUNBOOK, encoding="utf-8")


def _write_launcher() -> None:
    path = ROOT / "launch_after_scenario_envelope_thesis_ablation.sh"
    path.write_text(LAUNCHER, encoding="utf-8")
    path.chmod(0o755)


def _read_config_list(path: Path) -> list[Path]:
    if not path.exists():
        raise RuntimeError(f"missing config list: {path}")
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_manifest() -> dict[str, Any]:
    path = ROOT / "manifest.json"
    if not path.exists():
        raise RuntimeError(f"missing manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid manifest: {path}")
    return payload


def _sanity_check(
    full_configs: list[Path],
    smoke_configs: list[Path],
    manifest: dict[str, Any],
    fallback_plan: dict[str, dict[str, Any]],
) -> None:
    if len(full_configs) != EXPECTED_FULL:
        raise RuntimeError(f"expected {EXPECTED_FULL} full configs, found {len(full_configs)}")
    if len(smoke_configs) != EXPECTED_SMOKE:
        raise RuntimeError(f"expected {EXPECTED_SMOKE} smoke configs, found {len(smoke_configs)}")
    payloads = [_load_config(path) for path in full_configs]
    smoke_payloads = [_load_config(path) for path in smoke_configs]
    run_ids = [str(payload["run_id"]) for payload in payloads + smoke_payloads]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError("duplicate run_id")

    all_payloads = payloads + smoke_payloads
    if {payload["agent_count"] for payload in all_payloads} != {6}:
        raise RuntimeError("all configs must be s06")
    if {payload["persona_profile"] for payload in all_payloads} != {"action_core_8"}:
        raise RuntimeError("all configs must use action_core_8")
    if {payload["asset_simulator_mode"] for payload in all_payloads} != {"scenario_envelope"}:
        raise RuntimeError("all configs must use scenario_envelope")
    if {payload["tool_policy"] for payload in all_payloads} != {"asset_simulator_v1"}:
        raise RuntimeError("all configs must use asset_simulator_v1")
    if {payload["candidate_sizing_mode"] for payload in all_payloads} != {"large"}:
        raise RuntimeError("all configs must use large candidate sizing")
    if {payload["candidate_sizing_max_candidates"] for payload in all_payloads} != {8}:
        raise RuntimeError("all configs must cap candidates at 8")
    if {payload["forecaster_routing_mode"] for payload in all_payloads} != {"run_level"}:
        raise RuntimeError("all configs must use run-level forecaster routing")
    if {payload["start_timestamp"] for payload in payloads} != set(WINDOWS.values()):
        raise RuntimeError("full configs must cover exactly the core windows")
    if {payload["ticks"] for payload in payloads} != {24}:
        raise RuntimeError("full configs must be 24 ticks")
    if {payload["ticks"] for payload in smoke_payloads} != {2}:
        raise RuntimeError("smoke configs must be 2 ticks")
    if set(fallback_plan) != set(INTENDED_FORECASTERS):
        raise RuntimeError("forecaster validation set drift")

    manifest_rows = manifest.get("full_runs", [])
    intended = {row.get("intended_forecaster") for row in manifest_rows}
    if intended != set(INTENDED_FORECASTERS):
        raise RuntimeError(f"unexpected intended forecasters: {sorted(intended)}")
    for row in manifest_rows + manifest.get("smoke_runs", []):
        intended_name = str(row.get("intended_forecaster"))
        actual = str(row.get("actual_forecaster"))
        record = fallback_plan[intended_name]
        if actual != record["actual_forecaster"]:
            raise RuntimeError(f"manifest/config fallback drift for {intended_name}")
        if bool(row.get("fallback_used")) != bool(record["fallback_used"]):
            raise RuntimeError(f"manifest fallback flag drift for {intended_name}")

    invariant_keys = {
        "run_id",
        "ticks",
        "start_timestamp",
        "forecaster_backend",
        "llm",
    }
    base = _without_keys(payloads[0], invariant_keys)
    for payload in payloads[1:]:
        comparable = _without_keys(payload, invariant_keys)
        if comparable != base:
            raise RuntimeError(f"unexpected non-axis drift in {payload['run_id']}")


def _without_keys(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    copied = deepcopy(payload)
    for key in keys:
        copied.pop(key, None)
    return copied


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
        raise RuntimeError("refusing duplicate completed s06-forecaster-breadth outputs:\n" + "\n".join(collisions))


RUNBOOK = """# S06 Forecaster Breadth 2026-05-20

This matrix chains after `scenario-envelope-thesis-ablation-20260520` and isolates forecaster choice for the s06 action-core society.

## Matrix

- 36 full runs: 12 representative forecasters x 3 core windows.
- 3 smoke runs: f0, f8, and f10 on `apr02-0530`.
- All runs use `asset_simulator_mode: scenario_envelope`, `tool_policy: asset_simulator_v1`, `forecaster_routing_mode: run_level`, and `candidate_sizing_mode: large`.
- Fallbacks are resolved before launch and recorded in `manifest.json`; run configs contain the actual backend that will run.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_s06_forecaster_breadth.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/s06-forecaster-breadth-20260520/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/s06-forecaster-breadth-20260520/all.txt
```

## Launch

```bash
cd /home/ucloud/heimdall
bash ai-society/configs/s06-forecaster-breadth-20260520/launch_after_scenario_envelope_thesis_ablation.sh
```
"""


LAUNCHER = """#!/usr/bin/env bash
set -euo pipefail

cd /home/ucloud/heimdall
export PYTHONPATH=.

BASE_URL="http://127.0.0.1:8000/v1"
UPSTREAM_ROOT="ai-society/configs/scenario-envelope-thesis-ablation-20260520"
UPSTREAM_EXPECTED=30

ROOT="ai-society/configs/s06-forecaster-breadth-20260520"
RUN_ROOT="ai-society/runs/s06-forecaster-breadth-20260520"
TARGET_SESSION="heimdall-s06-fb"
TARGET_STAGE="s06-forecaster-breadth"
TARGET_EXPECTED=36

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launch.lock"
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another s06-forecaster-breadth launcher owns the lock; exiting"
  exit 0
fi

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

stage_status() {
  local results="$1"
  local expected="$2"
  RESULTS="$results" EXPECTED="$expected" uv run python - <<'PY_STATUS'
import json
import os
from pathlib import Path

path = Path(os.environ["RESULTS"])
expected = int(os.environ["EXPECTED"])
rows = json.loads(path.read_text()) if path.exists() else []
failed = [row for row in rows if row.get("ok") is False]
ok = [row for row in rows if row.get("ok") is True]
if failed:
    print("failed")
elif len(ok) >= expected:
    print("complete")
else:
    print(f"running:{len(ok)}")
PY_STATUS
}

wait_for_upstream() {
  while true; do
    local status="missing-log-dir"
    local results=""
    if [ -f "$UPSTREAM_ROOT/latest-log-dir.txt" ]; then
      local log_dir
      log_dir="$(cat "$UPSTREAM_ROOT/latest-log-dir.txt")"
      results="$log_dir/gpu0-results.json"
      status="missing-results"
      if [ -f "$results" ]; then
        status="$(stage_status "$results" "$UPSTREAM_EXPECTED")"
        case "$status" in
          failed)
            echo "[$(timestamp)] upstream scenario-envelope-thesis-ablation failed; refusing s06 forecaster launch." >&2
            cat "$results" >&2
            exit 1
            ;;
          complete)
            echo "[$(timestamp)] upstream scenario-envelope-thesis-ablation complete with $UPSTREAM_EXPECTED successful rows"
            break
            ;;
        esac
      fi
    fi
    echo "[$(timestamp)] waiting for upstream scenario-envelope-thesis-ablation ($status)"
    sleep 300
  done
}

validate_configs() {
  uv run python tools/experiments/generate_s06_forecaster_breadth.py >/dev/null
  uv run python tools/experiments/generate_s06_forecaster_breadth.py --check-only >/dev/null
  while read -r cfg; do
    [ -z "$cfg" ] && continue
    uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
  done < "$ROOT/smoke.txt"
  while read -r cfg; do
    [ -z "$cfg" ] && continue
    uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
  done < "$ROOT/all.txt"
}

matching_runner_active() {
  pgrep -af "run_market_intelligence_stage.py --stage $TARGET_STAGE .*--log-dir $RUN_ROOT/logs-" >/dev/null
}

prepare_target_session() {
  if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
    if matching_runner_active; then
      echo "tmux session $TARGET_SESSION already runs $TARGET_STAGE; not launching duplicate."
      return 1
    fi
    echo "tmux session $TARGET_SESSION exists but no matching runner is active; removing stale session."
    tmux kill-session -t "=$TARGET_SESSION"
  fi
  return 0
}

run_smoke_stage() {
  local log_dir="$RUN_ROOT/logs-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s\n' "$log_dir" > "$ROOT/latest-smoke-log-dir.txt"
  echo "[$(timestamp)] smoke start $TARGET_STAGE log_dir=$log_dir"
  uv run python ai-society/run_market_intelligence_stage.py \\
    --stage "$TARGET_STAGE-smoke" \\
    --gpu gpu0 \\
    --base-url "$BASE_URL" \\
    --config-list "$ROOT/smoke.txt" \\
    --log-dir "$log_dir" \\
    > "$log_dir/sequential.stdout.log" 2>&1
  echo "[$(timestamp)] smoke complete $TARGET_STAGE"
}

launch_full_stage() {
  prepare_target_session || return 0
  local log_dir="$RUN_ROOT/logs-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s\n' "$log_dir" > "$ROOT/latest-log-dir.txt"
  tmux new-session -d -s "$TARGET_SESSION" "
cd /home/ucloud/heimdall &&
export PYTHONPATH=. &&
uv run python ai-society/run_market_intelligence_stage.py   --stage '$TARGET_STAGE'   --gpu gpu0   --base-url '$BASE_URL'   --config-list '$ROOT/all.txt'   --log-dir '$log_dir'   > '$log_dir/sequential.stdout.log' 2>&1
"
  verify_launch "$log_dir"
  wait_for_stage_results "$log_dir/gpu0-results.json"
}

verify_launch() {
  local log_dir="$1"
  for _ in $(seq 1 12); do
    if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null && matching_runner_active && grep -q "1/$TARGET_EXPECTED start" "$log_dir/sequential.stdout.log" 2>/dev/null; then
      echo "[$(timestamp)] verified $TARGET_SESSION started $TARGET_STAGE"
      return 0
    fi
    sleep 5
  done
  echo "Launch verification failed for $TARGET_SESSION." >&2
  tmux has-session -t "=$TARGET_SESSION" 2>/dev/null || echo "missing tmux session" >&2
  matching_runner_active || echo "missing matching runner process" >&2
  tail -80 "$log_dir/sequential.stdout.log" >&2 || true
  exit 1
}

wait_for_stage_results() {
  local results="$1"
  while true; do
    local status="missing-results"
    if [ -f "$results" ]; then
      status="$(stage_status "$results" "$TARGET_EXPECTED")"
      case "$status" in
        failed)
          echo "[$(timestamp)] s06-forecaster-breadth failed." >&2
          cat "$results" >&2
          exit 1
          ;;
        complete)
          echo "[$(timestamp)] s06-forecaster-breadth complete with $TARGET_EXPECTED successful rows"
          uv run python tools/evaluation/compare_s06_forecaster_breadth.py >/dev/null
          return 0
          ;;
      esac
    fi
    if ! tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
      echo "[$(timestamp)] $TARGET_SESSION is gone but $results does not show $TARGET_EXPECTED successful rows." >&2
      [ -f "$results" ] && cat "$results" >&2 || true
      exit 1
    fi
    echo "[$(timestamp)] waiting for $TARGET_SESSION ($status)"
    sleep 300
  done
}

wait_for_upstream
validate_configs
run_smoke_stage
launch_full_stage

echo "[$(timestamp)] s06-forecaster-breadth matrix complete"
"""


if __name__ == "__main__":
    main()
