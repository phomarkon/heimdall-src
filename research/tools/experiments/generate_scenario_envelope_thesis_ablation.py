from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("ai-society/configs/scenario-envelope-thesis-ablation-20260520")
RUN_ROOT = Path("ai-society/runs/scenario-envelope-thesis-ablation-20260520")
UPSTREAM_ROOT = Path("ai-society/configs/scenario-envelope-breadth-20260520")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"
MODEL = "Qwen/Qwen3-32B"

CORE_WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}
BREADTH_WINDOWS = {
    "apr06-1300": "2026-04-06T13:00:00Z",
    "apr22-0830": "2026-04-22T08:30:00Z",
    "apr27-1730": "2026-04-27T17:30:00Z",
}
SOCIETIES = {
    "s06-actioncore": {
        "agent_count": 6,
        "persona_profile": "action_core_8",
        "max_concurrency": 8,
        "per_endpoint_max_concurrency": 4,
    },
    "s12-balanced": {
        "agent_count": 12,
        "persona_profile": "balanced_intelligence",
        "max_concurrency": 12,
        "per_endpoint_max_concurrency": 6,
    },
    "s20-mixed": {
        "agent_count": 20,
        "persona_profile": "mixed_expert_20_sideaware",
        "max_concurrency": 16,
        "per_endpoint_max_concurrency": 8,
    },
}
DEFAULT_SEED = 42
ROBUSTNESS_SEEDS = (13, 137)
EXPECTED_FULL = 30
EXPECTED_SMOKE = 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or check scenario-envelope thesis ablation configs.")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        full = _read_config_list(ROOT / "all.txt")
        smoke = _read_config_list(ROOT / "smoke.txt")
        _sanity_check(full, smoke)
        _assert_no_existing_outputs({path.stem for path in full + smoke})
        _assert_no_current_matrix_duplicates(full)
        print(json.dumps({"ok": True, "full": len(full), "smoke": len(smoke), "config_root": str(ROOT)}, indent=2))
        return

    ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    full_configs: list[Path] = []
    smoke_configs: list[Path] = []
    seen: set[str] = set()

    all_windows = {**CORE_WINDOWS, **BREADTH_WINDOWS}
    for window_slug, timestamp in all_windows.items():
        for society_slug in SOCIETIES:
            run_id = f"seta-{society_slug}-scenario-large-{window_slug}-seed{DEFAULT_SEED}-24-q32"
            path = ROOT / "full" / "breadth" / society_slug / f"{run_id}.yaml"
            _write_config(
                path,
                _config(
                    run_id=run_id,
                    society_slug=society_slug,
                    timestamp=timestamp,
                    seed=DEFAULT_SEED,
                    ticks=24,
                ),
                seen,
            )
            full_configs.append(path)

    for window_slug, timestamp in CORE_WINDOWS.items():
        for society_slug in ("s06-actioncore", "s20-mixed"):
            for seed in ROBUSTNESS_SEEDS:
                run_id = f"seta-{society_slug}-scenario-large-{window_slug}-seed{seed}-24-q32"
                path = ROOT / "full" / "seed-robustness" / society_slug / f"{run_id}.yaml"
                _write_config(
                    path,
                    _config(
                        run_id=run_id,
                        society_slug=society_slug,
                        timestamp=timestamp,
                        seed=seed,
                        ticks=24,
                    ),
                    seen,
                )
                full_configs.append(path)

    smoke_specs = [
        ("s06-actioncore", "apr02-0530", DEFAULT_SEED),
        ("s20-mixed", "apr02-0530", 13),
    ]
    for society_slug, window_slug, seed in smoke_specs:
        run_id = f"smoke-seta-{society_slug}-scenario-large-{window_slug}-seed{seed}-2-q32"
        path = ROOT / "smoke" / society_slug / f"{run_id}.yaml"
        _write_config(
            path,
            _config(
                run_id=run_id,
                society_slug=society_slug,
                timestamp=CORE_WINDOWS[window_slug],
                seed=seed,
                ticks=2,
            ),
            seen,
        )
        smoke_configs.append(path)

    _assert_no_existing_outputs({path.stem for path in full_configs + smoke_configs})
    _assert_no_current_matrix_duplicates(full_configs)
    _sanity_check(full_configs, smoke_configs)
    _write_lists(full_configs, smoke_configs)
    _write_manifest(full_configs, smoke_configs)
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
            },
            indent=2,
        )
    )


def _config(
    *,
    run_id: str,
    society_slug: str,
    timestamp: str,
    seed: int,
    ticks: int,
) -> dict[str, Any]:
    society = SOCIETIES[society_slug]
    return {
        "run_id": run_id,
        "seed": seed,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": society["agent_count"],
        "ticks": ticks,
        "start_timestamp": timestamp,
        "forecaster_backend": "f8",
        "forecaster_routing_mode": "persona",
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
        "persona_profile": society["persona_profile"],
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
            "max_concurrency": society["max_concurrency"],
            "per_endpoint_max_concurrency": society["per_endpoint_max_concurrency"],
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


def _write_manifest(full_configs: list[Path], smoke_configs: list[Path]) -> None:
    payload = {
        "matrix": "scenario-envelope-thesis-ablation-20260520",
        "generated_at": "2026-05-20",
        "full_run_count": len(full_configs),
        "smoke_run_count": len(smoke_configs),
        "default_seed": DEFAULT_SEED,
        "robustness_seeds": list(ROBUSTNESS_SEEDS),
        "core_windows": CORE_WINDOWS,
        "breadth_windows": BREADTH_WINDOWS,
        "societies": SOCIETIES,
        "asset_simulator_mode": "scenario_envelope",
        "candidate_sizing_mode": "large",
        "chain_after": {
            "matrix": "scenario-envelope-breadth-20260520",
            "required_completed": 26,
            "required_failed": 0,
        },
        "config_list": str(ROOT / "all.txt"),
        "smoke_list": str(ROOT / "smoke.txt"),
        "run_root": str(RUN_ROOT),
        "full_runs": [_run_manifest_row(path) for path in full_configs],
        "smoke_runs": [_run_manifest_row(path) for path in smoke_configs],
    }
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_manifest_row(path: Path) -> dict[str, Any]:
    payload = _load_config(path)
    return {
        "run_id": path.stem,
        "config": str(path),
        "agent_count": payload["agent_count"],
        "ticks": payload["ticks"],
        "start_timestamp": payload["start_timestamp"],
        "seed": payload["seed"],
        "asset_simulator_mode": payload["asset_simulator_mode"],
        "candidate_sizing_mode": payload["candidate_sizing_mode"],
        "persona_profile": payload["persona_profile"],
    }


def _write_runbook() -> None:
    (ROOT / "RUNBOOK.md").write_text(RUNBOOK, encoding="utf-8")


def _write_launcher() -> None:
    path = ROOT / "launch_after_scenario_envelope_breadth.sh"
    path.write_text(LAUNCHER, encoding="utf-8")
    path.chmod(0o755)


def _read_config_list(path: Path) -> list[Path]:
    if not path.exists():
        raise RuntimeError(f"missing config list: {path}")
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sanity_check(full_configs: list[Path], smoke_configs: list[Path]) -> None:
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
    if {payload["asset_simulator_mode"] for payload in all_payloads} != {"scenario_envelope"}:
        raise RuntimeError("all configs must use scenario_envelope")
    if {payload["tool_policy"] for payload in all_payloads} != {"asset_simulator_v1"}:
        raise RuntimeError("all configs must use asset_simulator_v1")
    if {payload["preprobe_mode"] for payload in all_payloads} != {"full"}:
        raise RuntimeError("all configs must use full preprobe")
    if {payload["candidate_sizing_mode"] for payload in all_payloads} != {"large"}:
        raise RuntimeError("all configs must use large candidate sizing")
    if {payload["forecaster_routing_mode"] for payload in all_payloads} != {"persona"}:
        raise RuntimeError("all configs must use persona forecaster routing")

    breadth = [
        payload
        for payload in payloads
        if payload["seed"] == DEFAULT_SEED
        and payload["start_timestamp"] in {*CORE_WINDOWS.values(), *BREADTH_WINDOWS.values()}
    ]
    if len(breadth) != 18:
        raise RuntimeError(f"expected 18 seed42 breadth configs, found {len(breadth)}")
    robustness = [payload for payload in payloads if payload["seed"] in ROBUSTNESS_SEEDS]
    if len(robustness) != 12:
        raise RuntimeError(f"expected 12 seed robustness configs, found {len(robustness)}")
    if {payload["start_timestamp"] for payload in robustness} != set(CORE_WINDOWS.values()):
        raise RuntimeError("seed robustness must cover core windows only")
    if {payload["persona_profile"] for payload in robustness} != {
        "action_core_8",
        "mixed_expert_20_sideaware",
    }:
        raise RuntimeError("seed robustness must cover s06 and s20 only")

    invariant_keys = {
        "run_id",
        "seed",
        "agent_count",
        "ticks",
        "start_timestamp",
        "persona_profile",
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
        raise RuntimeError("refusing duplicate completed scenario-envelope-thesis-ablation outputs:\n" + "\n".join(collisions))


def _assert_no_current_matrix_duplicates(configs: list[Path]) -> None:
    if not UPSTREAM_ROOT.exists():
        return
    current = [path for path in UPSTREAM_ROOT.rglob("*.yaml")]
    planned_payloads = [_load_config(path) for path in configs]
    current_payloads = [_load_config(path) for path in current]
    duplicates = []
    for planned in planned_payloads:
        for existing in current_payloads:
            if _duplicate_key(planned) == _duplicate_key(existing):
                duplicates.append((planned["run_id"], existing["run_id"]))
    if duplicates:
        lines = [f"{planned} duplicates current matrix run {existing}" for planned, existing in duplicates]
        raise RuntimeError("refusing current-matrix duplicates:\n" + "\n".join(lines))


def _duplicate_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    mode = payload.get("asset_simulator_mode")
    if mode in {"real", "dual_compare_real_controls"}:
        mode = "scenario_envelope"
    return (
        payload.get("agent_count"),
        payload.get("persona_profile"),
        payload.get("start_timestamp"),
        payload.get("ticks"),
        payload.get("seed"),
        mode,
        payload.get("candidate_sizing_mode"),
        payload.get("tool_policy"),
        payload.get("preprobe_mode"),
    )


RUNBOOK = """# Scenario Envelope Thesis Ablation 2026-05-20

This matrix chains after `scenario-envelope-breadth-20260520` and fills the clean thesis comparison grid for s06, s12, and mixed20 societies.

## Matrix

- 18 breadth runs: s06-actioncore, s12-balanced, and s20-mixed x six April windows x seed 42.
- 12 seed robustness runs: s06-actioncore and s20-mixed x three core windows x seeds 13 and 137.
- 2 smoke runs on `apr02-0530`.
- All runs use `asset_simulator_mode: scenario_envelope`, `tool_policy: asset_simulator_v1`, `preprobe_mode: full`, and `candidate_sizing_mode: large`.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_scenario_envelope_thesis_ablation.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/scenario-envelope-thesis-ablation-20260520/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/scenario-envelope-thesis-ablation-20260520/all.txt
```

## Launch

```bash
cd /home/ucloud/heimdall
bash ai-society/configs/scenario-envelope-thesis-ablation-20260520/launch_after_scenario_envelope_breadth.sh
```
"""


LAUNCHER = """#!/usr/bin/env bash
set -euo pipefail

cd /home/ucloud/heimdall
export PYTHONPATH=.

BASE_URL="http://127.0.0.1:8000/v1"
UPSTREAM_ROOT="ai-society/configs/scenario-envelope-breadth-20260520"
UPSTREAM_EXPECTED=26

ROOT="ai-society/configs/scenario-envelope-thesis-ablation-20260520"
RUN_ROOT="ai-society/runs/scenario-envelope-thesis-ablation-20260520"
TARGET_SESSION="heimdall-seta"
TARGET_STAGE="scenario-envelope-thesis-ablation"
TARGET_EXPECTED=30

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launch.lock"
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another scenario-envelope-thesis-ablation launcher owns the lock; exiting"
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
            echo "[$(timestamp)] upstream scenario-envelope-breadth failed; refusing thesis ablation launch." >&2
            cat "$results" >&2
            exit 1
            ;;
          complete)
            echo "[$(timestamp)] upstream scenario-envelope-breadth complete with $UPSTREAM_EXPECTED successful rows"
            break
            ;;
        esac
      fi
    fi
    echo "[$(timestamp)] waiting for upstream scenario-envelope-breadth ($status)"
    sleep 300
  done
}

validate_configs() {
  uv run python tools/experiments/generate_scenario_envelope_thesis_ablation.py --check-only >/dev/null
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
  uv run python ai-society/run_market_intelligence_stage.py \
    --stage "$TARGET_STAGE-smoke" \
    --gpu gpu0 \
    --base-url "$BASE_URL" \
    --config-list "$ROOT/smoke.txt" \
    --log-dir "$log_dir" \
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
uv run python ai-society/run_market_intelligence_stage.py \
  --stage '$TARGET_STAGE' \
  --gpu gpu0 \
  --base-url '$BASE_URL' \
  --config-list '$ROOT/all.txt' \
  --log-dir '$log_dir' \
  > '$log_dir/sequential.stdout.log' 2>&1
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
          echo "[$(timestamp)] scenario-envelope-thesis-ablation failed." >&2
          cat "$results" >&2
          exit 1
          ;;
        complete)
          echo "[$(timestamp)] scenario-envelope-thesis-ablation complete with $TARGET_EXPECTED successful rows"
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

echo "[$(timestamp)] scenario-envelope-thesis-ablation matrix complete"
"""


if __name__ == "__main__":
    main()
