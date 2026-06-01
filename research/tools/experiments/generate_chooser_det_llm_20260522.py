from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


MATRIX = "chooser-det-llm-20260522"
ROOT = Path(f"ai-society/configs/{MATRIX}")
RUN_ROOT = Path(f"ai-society/runs/{MATRIX}")
CONTEXT_DIR = "data/cache/real_context/april_2026"
TRUTH_DIR = "data/cache/evaluation_truth/april_2026"
MEMORY_BANK = "ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl"
MODEL = "Qwen/Qwen3-32B"
DEFAULT_SEED = 42
EXPECTED_FULL = 45
EXPECTED_SMOKE = 3

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
    "apr17-1900": "2026-04-17T19:00:00Z",
    "apr28-1900": "2026-04-28T19:00:00Z",
}
SOCIETIES = {
    "s06-actioncore": {
        "agent_count": 6,
        "persona_profile": "action_core_8",
    },
    "s12-balanced": {
        "agent_count": 12,
        "persona_profile": "balanced_intelligence",
    },
    "mixed20": {
        "agent_count": 20,
        "persona_profile": "mixed_expert_20_sideaware",
    },
}
VARIANTS = {
    "deterministic": {
        "chooser_mode": "deterministic_best_accepted",
        "final_bid_guard": "schema_only_shadow",
        "safety_toolset": "full",
        "llm_enabled": False,
        "temperature": 0.0,
        "max_tokens": 512,
    },
    "guarded": {
        "chooser_mode": "llm",
        "final_bid_guard": "simulator_exact_match",
        "safety_toolset": "full",
        "llm_enabled": True,
        "temperature": 0.2,
        "max_tokens": 640,
    },
    "shadow-toolvisible": {
        "chooser_mode": "llm",
        "final_bid_guard": "schema_only_shadow",
        "safety_toolset": "full",
        "llm_enabled": True,
        "temperature": 0.2,
        "max_tokens": 640,
    },
}


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
    smoke_configs: list[Path] = []
    seen: set[str] = set()

    for window_slug, timestamp in WINDOWS.items():
        for society_slug in SOCIETIES:
            for variant_slug in VARIANTS:
                run_id = f"cdl-{society_slug}-{variant_slug}-{window_slug}-seed{DEFAULT_SEED}-24-q32"
                path = ROOT / "full" / society_slug / variant_slug / f"{run_id}.yaml"
                _write_config(
                    path,
                    _config(
                        run_id=run_id,
                        society_slug=society_slug,
                        variant_slug=variant_slug,
                        timestamp=timestamp,
                        ticks=24,
                    ),
                    seen,
                )
                full_configs.append(path)

    for variant_slug in VARIANTS:
        run_id = f"smoke-cdl-s06-actioncore-{variant_slug}-apr02-0530-seed{DEFAULT_SEED}-2-q32"
        path = ROOT / "smoke" / "s06-actioncore" / variant_slug / f"{run_id}.yaml"
        _write_config(
            path,
            _config(
                run_id=run_id,
                society_slug="s06-actioncore",
                variant_slug=variant_slug,
                timestamp=WINDOWS["apr02-0530"],
                ticks=2,
            ),
            seen,
        )
        smoke_configs.append(path)

    _assert_no_existing_outputs({path.stem for path in full_configs + smoke_configs})
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
    variant_slug: str,
    timestamp: str,
    ticks: int,
) -> dict[str, Any]:
    society = SOCIETIES[society_slug]
    variant = VARIANTS[variant_slug]
    return {
        "run_id": run_id,
        "seed": DEFAULT_SEED,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": society["agent_count"],
        "ticks": ticks,
        "start_timestamp": timestamp,
        "forecaster_backend": "f8",
        "forecaster_routing_mode": "persona",
        "chooser_mode": variant["chooser_mode"],
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "preprobe_mode": "full",
        "objective": "bid_seeking",
        "final_bid_guard": variant["final_bid_guard"],
        "safety_toolset": variant["safety_toolset"],
        "ablation_strategy": "comm_broadcast_digest_priority_calibration",
        "persona_profile": society["persona_profile"],
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "asset_simulator_v1",
        "asset_simulator_mode": "scenario_envelope",
        "asset_proxy_style": "market",
        "candidate_sizing_mode": "medium",
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
            "enabled": variant["llm_enabled"],
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "api_key": "heimdall-local",
            "model": MODEL,
            "temperature": variant["temperature"],
            "max_tokens": variant["max_tokens"],
            "timeout_seconds": 180,
            "max_concurrency": 12,
            "per_endpoint_max_concurrency": 6,
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
        "matrix": MATRIX,
        "generated_at": "2026-05-22",
        "question": "Does the LLM genuinely add value beyond a greedy deterministic selector, or does all value come from simulator data availability?",
        "full_run_count": len(full_configs),
        "smoke_run_count": len(smoke_configs),
        "seed": DEFAULT_SEED,
        "forecaster_backend": "f8",
        "forecaster_routing_mode": "persona",
        "asset_simulator_mode": "scenario_envelope",
        "tool_policy": "asset_simulator_v1",
        "candidate_sizing_mode": "medium",
        "windows": WINDOWS,
        "societies": SOCIETIES,
        "variants": VARIANTS,
        "chain_after": {
            "matrix": "deliberation-s06-scenario-large-20260521",
            "tmux_session": "heimdall-delib-s06-large-20260521",
            "required_completed": 3,
            "required_failed": 0,
        },
        "truth_dir": TRUTH_DIR,
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
        "chooser_mode": payload["chooser_mode"],
        "final_bid_guard": payload["final_bid_guard"],
        "llm_enabled": payload["llm"]["enabled"],
        "persona_profile": payload["persona_profile"],
        "start_timestamp": payload["start_timestamp"],
        "ticks": payload["ticks"],
    }


def _write_runbook() -> None:
    (ROOT / "RUNBOOK.md").write_text(RUNBOOK, encoding="utf-8")


def _write_launcher() -> None:
    path = ROOT / "launch_after_current_matrix.sh"
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
    all_payloads = payloads + smoke_payloads
    run_ids = [str(payload["run_id"]) for payload in all_payloads]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError("duplicate run_id")

    _expect_values(all_payloads, "seed", {DEFAULT_SEED})
    _expect_values(all_payloads, "forecaster_seed", {42})
    _expect_values(all_payloads, "forecaster_backend", {"f8"})
    _expect_values(all_payloads, "forecaster_routing_mode", {"persona"})
    _expect_values(all_payloads, "asset_simulator_mode", {"scenario_envelope"})
    _expect_values(all_payloads, "tool_policy", {"asset_simulator_v1"})
    _expect_values(all_payloads, "preprobe_mode", {"full"})
    _expect_values(all_payloads, "ablation_strategy", {"comm_broadcast_digest_priority_calibration"})
    _expect_values(all_payloads, "candidate_sizing_mode", {"medium"})

    full_keys = {
        (
            payload["chooser_mode"],
            payload["final_bid_guard"],
            payload["llm"]["enabled"],
            payload["persona_profile"],
            payload["start_timestamp"],
        )
        for payload in payloads
    }
    if len(full_keys) != EXPECTED_FULL:
        raise RuntimeError("full matrix does not cover unique chooser/society/window cells")

    smoke_keys = {(payload["chooser_mode"], payload["final_bid_guard"], payload["llm"]["enabled"]) for payload in smoke_payloads}
    if len(smoke_keys) != EXPECTED_SMOKE:
        raise RuntimeError("smoke matrix does not cover the three chooser variants")
    if {payload["ticks"] for payload in smoke_payloads} != {2}:
        raise RuntimeError("smoke configs must be 2 ticks")
    if {payload["ticks"] for payload in payloads} != {24}:
        raise RuntimeError("full configs must be 24 ticks")
    if {payload["agent_count"] for payload in smoke_payloads} != {6}:
        raise RuntimeError("smoke configs must be s06 only")
    for payload in all_payloads:
        _validate_variant_payload(payload)
        if payload["llm"]["model"] != MODEL:
            raise RuntimeError(f"{payload['run_id']} must keep llm.model={MODEL}")
        if payload["llm"]["base_urls"] != ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"]:
            raise RuntimeError(f"{payload['run_id']} must use both vLLM endpoints")
        if payload["llm"]["max_concurrency"] != 12 or payload["llm"]["per_endpoint_max_concurrency"] != 6:
            raise RuntimeError(f"{payload['run_id']} has bad LLM concurrency")

    invariant_keys = {
        "run_id",
        "agent_count",
        "ticks",
        "start_timestamp",
        "chooser_mode",
        "final_bid_guard",
        "safety_toolset",
        "persona_profile",
        "llm",
    }
    base = _without_keys(payloads[0], invariant_keys)
    for payload in payloads[1:]:
        comparable = _without_keys(payload, invariant_keys)
        if comparable != base:
            raise RuntimeError(f"unexpected non-axis drift in {payload['run_id']}")


def _validate_variant_payload(payload: dict[str, Any]) -> None:
    if payload["chooser_mode"] == "deterministic_best_accepted":
        if payload["final_bid_guard"] != "schema_only_shadow" or payload["llm"]["enabled"] is not False:
            raise RuntimeError(f"bad deterministic config: {payload['run_id']}")
        return
    if payload["chooser_mode"] != "llm" or payload["llm"]["enabled"] is not True:
        raise RuntimeError(f"bad chooser config: {payload['run_id']}")
    if payload["final_bid_guard"] not in {"simulator_exact_match", "schema_only_shadow"}:
        raise RuntimeError(f"bad final_bid_guard: {payload['run_id']}")


def _expect_values(payloads: list[dict[str, Any]], key: str, expected: set[Any]) -> None:
    values = {payload[key] for payload in payloads}
    if values != expected:
        raise RuntimeError(f"{key} expected {expected}, found {values}")


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
        raise RuntimeError("refusing duplicate completed chooser-det-llm outputs:\n" + "\n".join(collisions))


RUNBOOK = """# Chooser Deterministic vs LLM 2026-05-22

This matrix compares whether Qwen3-32B adds value beyond greedy deterministic selection when simulator evidence is held constant.

## Matrix

- 45 full runs: deterministic, guarded LLM, and shadow-toolvisible LLM across three societies and five windows.
- 3 smoke runs: s06-actioncore on apr02-0530, one per chooser variant.
- All runs use `asset_simulator_mode: scenario_envelope`, `tool_policy: asset_simulator_v1`, `preprobe_mode: full`, `candidate_sizing_mode: medium`, `forecaster_backend: f8`, and seed 42.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_chooser_det_llm_20260522.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/chooser-det-llm-20260522/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/chooser-det-llm-20260522/all.txt
```

## Launch

```bash
cd /home/ucloud/heimdall
bash ai-society/configs/chooser-det-llm-20260522/launch_after_current_matrix.sh
```
"""


LAUNCHER = """#!/usr/bin/env bash
set -euo pipefail

cd /home/ucloud/heimdall
export PYTHONPATH=.

ROOT="ai-society/configs/chooser-det-llm-20260522"
RUN_ROOT="ai-society/runs/chooser-det-llm-20260522"
UPSTREAM_SESSION="heimdall-delib-s06-large-20260521"
UPSTREAM_RUN_ROOT="ai-society/runs/deliberation-s06-scenario-large-20260521"
UPSTREAM_EXPECTED=3
TARGET_SESSION="heimdall-chooser-det-llm"
TARGET_STAGE="chooser-det-llm-20260522"
TARGET_EXPECTED=45

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launch.lock"
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another chooser-det-llm launcher owns the lock; exiting"
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

latest_upstream_results() {
  find "$UPSTREAM_RUN_ROOT"/logs -path '*/results.json' -type f -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

wait_for_upstream() {
  while true; do
    local results
    results="$(latest_upstream_results || true)"
    local status="missing-results"
    if [ -n "$results" ] && [ -f "$results" ]; then
      status="$(stage_status "$results" "$UPSTREAM_EXPECTED")"
      case "$status" in
        failed)
          echo "[$(timestamp)] upstream deliberation matrix failed; refusing chooser-det-llm launch." >&2
          cat "$results" >&2
          exit 1
          ;;
        complete)
          echo "[$(timestamp)] upstream deliberation matrix complete with $UPSTREAM_EXPECTED successful rows"
          break
          ;;
      esac
    fi
    if ! tmux has-session -t "=$UPSTREAM_SESSION" 2>/dev/null && ! pgrep -af "run_long_model_society_matrix.py --config-list ai-society/configs/deliberation-s06-scenario-large-20260521/config-list.txt" >/dev/null; then
      echo "[$(timestamp)] upstream is not running and did not complete cleanly ($status)." >&2
      [ -n "$results" ] && [ -f "$results" ] && cat "$results" >&2 || true
      exit 1
    fi
    echo "[$(timestamp)] waiting for upstream deliberation matrix ($status)"
    sleep 300
  done
}

validate_configs() {
  uv run python tools/experiments/generate_chooser_det_llm_20260522.py --check-only >/dev/null
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
  pgrep -af "run_long_model_society_matrix.py --config-list $ROOT/all.txt" >/dev/null
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
  printf '%s\\n' "$log_dir" > "$ROOT/latest-smoke-log-dir.txt"
  echo "[$(timestamp)] smoke start $TARGET_STAGE log_dir=$log_dir"
  uv run python ai-society/run_long_model_society_matrix.py \
    --config-list "$ROOT/smoke.txt" \
    --log-dir "$log_dir" \
    --continue-on-failure \
    --skip-vllm-restart \
    > "$log_dir/controller.stdout.log" 2>&1
  local status
  status="$(stage_status "$log_dir/results.json" 3)"
  if [ "$status" != "complete" ]; then
    echo "[$(timestamp)] smoke failed or incomplete: $status" >&2
    cat "$log_dir/results.json" >&2 || true
    exit 1
  fi
  echo "[$(timestamp)] smoke complete $TARGET_STAGE"
}

launch_full_stage() {
  prepare_target_session || return 0
  local log_dir="$RUN_ROOT/logs-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s\\n' "$log_dir" > "$ROOT/latest-log-dir.txt"
  tmux new-session -d -s "$TARGET_SESSION" "
cd /home/ucloud/heimdall &&
export PYTHONPATH=. &&
uv run python ai-society/run_long_model_society_matrix.py \
  --config-list '$ROOT/all.txt' \
  --log-dir '$log_dir' \
  --continue-on-failure \
  --skip-vllm-restart \
  > '$log_dir/controller.stdout.log' 2>&1
"
  verify_launch "$log_dir"
  echo "[$(timestamp)] launched $TARGET_SESSION for $TARGET_STAGE log_dir=$log_dir"
}

verify_launch() {
  local log_dir="$1"
  for _ in $(seq 1 12); do
    if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null && matching_runner_active && grep -q "1/$TARGET_EXPECTED start" "$log_dir/controller.stdout.log" 2>/dev/null; then
      echo "[$(timestamp)] verified $TARGET_SESSION started $TARGET_STAGE"
      return 0
    fi
    sleep 5
  done
  echo "Launch verification failed for $TARGET_SESSION." >&2
  tmux has-session -t "=$TARGET_SESSION" 2>/dev/null || echo "missing tmux session" >&2
  matching_runner_active || echo "missing matching runner process" >&2
  tail -80 "$log_dir/controller.stdout.log" >&2 || true
  exit 1
}

wait_for_upstream
validate_configs
run_smoke_stage
launch_full_stage
"""


if __name__ == "__main__":
    main()
