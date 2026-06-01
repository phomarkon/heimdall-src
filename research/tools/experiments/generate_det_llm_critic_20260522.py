from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

MATRIX = "det-llm-critic-20260522"
ROOT = Path(f"ai-society/configs/{MATRIX}")
RUN_ROOT = Path(f"ai-society/runs/{MATRIX}")
CONTEXT_DIR = "data/cache/real_context/april_2026"
MODEL = "Qwen/Qwen3-32B"
SEED = 42
EXPECTED_FULL = 3
EXPECTED_SMOKE = 3

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}
SOCIETY = {
    "slug": "s06-actioncore",
    "agent_count": 6,
    "persona_profile": "action_core_8",
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
        run_id = f"dlc-s06-actioncore-critic-{window_slug}-seed{SEED}-24-q32"
        path = ROOT / "full" / "s06-actioncore" / f"{run_id}.yaml"
        _write_config(path, _config(run_id=run_id, timestamp=timestamp, ticks=24), seen)
        full_configs.append(path)

        smoke_id = f"smoke-dlc-s06-actioncore-critic-{window_slug}-seed{SEED}-2-q32"
        smoke_path = ROOT / "smoke" / "s06-actioncore" / f"{smoke_id}.yaml"
        _write_config(smoke_path, _config(run_id=smoke_id, timestamp=timestamp, ticks=2), seen)
        smoke_configs.append(smoke_path)

    _sanity_check(full_configs, smoke_configs)
    _assert_no_existing_outputs({path.stem for path in full_configs + smoke_configs})
    _write_lists(full_configs, smoke_configs)
    _write_manifest(full_configs, smoke_configs)
    _write_runbook()
    _write_launcher()
    print(json.dumps({"ok": True, "full_run_count": len(full_configs), "smoke_run_count": len(smoke_configs), "config_root": str(ROOT), "run_root": str(RUN_ROOT)}, indent=2))


def _config(*, run_id: str, timestamp: str, ticks: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": SEED,
        "forecaster_seed": SEED,
        "zone": "DK1",
        "agent_count": SOCIETY["agent_count"],
        "ticks": ticks,
        "start_timestamp": timestamp,
        "forecaster_backend": "f8",
        "forecaster_routing_mode": "persona",
        "chooser_mode": "deterministic_llm_critic",
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "preprobe_mode": "full",
        "objective": "bid_seeking",
        "final_bid_guard": "simulator_exact_match",
        "safety_toolset": "full",
        "ablation_strategy": "comm_broadcast_digest_priority_calibration",
        "persona_profile": SOCIETY["persona_profile"],
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
        "reviewer_mode": "code_only",
        "llm": {
            "enabled": True,
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "api_key": "heimdall-local",
            "model": MODEL,
            "temperature": 0.1,
            "max_tokens": 640,
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
        "question": "Can a forecast-diverse LLM critic reduce wrong-side and low-clearability deterministic bids without mutating simulator-backed candidates?",
        "full_run_count": len(full_configs),
        "smoke_run_count": len(smoke_configs),
        "scope": "s06-only first pass",
        "seed": SEED,
        "forecaster_backend": "f8",
        "forecaster_routing_mode": "persona",
        "chooser_mode": "deterministic_llm_critic",
        "final_bid_guard": "simulator_exact_match",
        "society": SOCIETY,
        "windows": WINDOWS,
        "chain_after": {
            "matrix": "chooser-det-llm-20260522",
            "required_completed": 45,
            "required_failed": 0,
        },
        "config_list": str(ROOT / "all.txt"),
        "smoke_list": str(ROOT / "smoke.txt"),
        "run_root": str(RUN_ROOT),
        "full_runs": [_row(path) for path in full_configs],
        "smoke_runs": [_row(path) for path in smoke_configs],
    }
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row(path: Path) -> dict[str, Any]:
    payload = _load_config(path)
    return {
        "run_id": payload["run_id"],
        "config": str(path),
        "agent_count": payload["agent_count"],
        "persona_profile": payload["persona_profile"],
        "start_timestamp": payload["start_timestamp"],
        "ticks": payload["ticks"],
        "chooser_mode": payload["chooser_mode"],
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
    payloads = [_load_config(path) for path in full_configs + smoke_configs]
    run_ids = [payload["run_id"] for payload in payloads]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError("duplicate run_id")
    for payload in payloads:
        expected = {
            "forecaster_backend": "f8",
            "forecaster_routing_mode": "persona",
            "chooser_mode": "deterministic_llm_critic",
            "asset_simulator_mode": "scenario_envelope",
            "tool_policy": "asset_simulator_v1",
            "preprobe_mode": "full",
            "ablation_strategy": "comm_broadcast_digest_priority_calibration",
            "final_bid_guard": "simulator_exact_match",
            "candidate_sizing_mode": "medium",
            "agent_count": 6,
            "persona_profile": "action_core_8",
        }
        for key, value in expected.items():
            if payload[key] != value:
                raise RuntimeError(f"{payload['run_id']} bad {key}: {payload[key]!r}")
        if payload["llm"]["enabled"] is not True or payload["llm"]["model"] != MODEL:
            raise RuntimeError(f"{payload['run_id']} has bad LLM config")
        if payload["llm"]["base_urls"] != ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"]:
            raise RuntimeError(f"{payload['run_id']} must use dual endpoints")
    if {payload["ticks"] for payload in _load_config_many(full_configs)} != {24}:
        raise RuntimeError("full configs must be 24 ticks")
    if {payload["ticks"] for payload in _load_config_many(smoke_configs)} != {2}:
        raise RuntimeError("smoke configs must be 2 ticks")


def _load_config_many(paths: list[Path]) -> list[dict[str, Any]]:
    return [_load_config(path) for path in paths]


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid config payload: {path}")
    return payload


def _assert_no_existing_outputs(run_ids: set[str]) -> None:
    collisions = [str(RUN_ROOT / run_id / "summary.json") for run_id in sorted(run_ids) if (RUN_ROOT / run_id / "summary.json").exists()]
    if collisions:
        raise RuntimeError("refusing duplicate completed det-llm-critic outputs:\n" + "\n".join(collisions))


RUNBOOK = """# Deterministic LLM Critic 2026-05-22

S06-only first pass for a forecast-diverse LLM critic. The deterministic proposer still selects exact simulator-backed candidates; Qwen3-32B can only keep the bid or veto it to watch/abstain.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_det_llm_critic_20260522.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/det-llm-critic-20260522/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/det-llm-critic-20260522/all.txt
```

## Launch

```bash
bash ai-society/configs/det-llm-critic-20260522/launch_after_current_matrix.sh
```
"""


LAUNCHER = """#!/usr/bin/env bash
set -euo pipefail

cd /home/ucloud/heimdall
export PYTHONPATH=.

ROOT="ai-society/configs/det-llm-critic-20260522"
RUN_ROOT="ai-society/runs/det-llm-critic-20260522"
UPSTREAM_RUN_ROOT="ai-society/runs/chooser-det-llm-20260522"
UPSTREAM_EXPECTED=45
TARGET_SESSION="heimdall-det-llm-critic-s06"
TARGET_EXPECTED=3

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launch.lock"
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another det-llm-critic launcher owns the lock; exiting"
  exit 0
fi

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

stage_status() {
  local results="$1"
  local expected="$2"
  RESULTS="$results" EXPECTED="$expected" uv run python - <<'PY_STATUS'
import json
import os
from pathlib import Path
rows = json.loads(Path(os.environ["RESULTS"]).read_text()) if Path(os.environ["RESULTS"]).exists() else []
failed = [row for row in rows if row.get("ok") is False]
ok = [row for row in rows if row.get("ok") is True]
if failed:
    print("failed")
elif len(ok) >= int(os.environ["EXPECTED"]):
    print("complete")
else:
    print(f"running:{len(ok)}")
PY_STATUS
}

latest_upstream_results() {
  find "$UPSTREAM_RUN_ROOT"/logs-* -name results.json -type f -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
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
          echo "[$(timestamp)] upstream CDL matrix failed; refusing det-llm-critic launch." >&2
          cat "$results" >&2
          exit 1
          ;;
        complete)
          echo "[$(timestamp)] upstream CDL matrix complete with $UPSTREAM_EXPECTED successful rows"
          break
          ;;
      esac
    fi
    if ! tmux has-session -t "=heimdall-chooser-det-llm" 2>/dev/null && ! pgrep -af "run_long_model_society_matrix.py --config-list ai-society/configs/chooser-det-llm-20260522/all.txt" >/dev/null; then
      echo "[$(timestamp)] upstream CDL is not running and did not complete cleanly ($status)." >&2
      [ -n "$results" ] && [ -f "$results" ] && cat "$results" >&2 || true
      exit 1
    fi
    echo "[$(timestamp)] waiting for upstream CDL matrix ($status)"
    sleep 300
  done
}

validate_configs() {
  uv run python tools/experiments/generate_det_llm_critic_20260522.py --check-only >/dev/null
  while read -r cfg; do [ -z "$cfg" ] || uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null; done < "$ROOT/smoke.txt"
  while read -r cfg; do [ -z "$cfg" ] || uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null; done < "$ROOT/all.txt"
}

run_smoke_stage() {
  local log_dir="$RUN_ROOT/logs-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s\\n' "$log_dir" > "$ROOT/latest-smoke-log-dir.txt"
  echo "[$(timestamp)] smoke start det-llm-critic-s06 log_dir=$log_dir"
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
}

launch_full_stage() {
  if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
    echo "tmux session $TARGET_SESSION already exists; not launching duplicate."
    return 0
  fi
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
  sleep 5
  if ! tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
    echo "[$(timestamp)] launch verification failed for $TARGET_SESSION" >&2
    tail -80 "$log_dir/controller.stdout.log" >&2 || true
    exit 1
  fi
  echo "[$(timestamp)] launched $TARGET_SESSION for 3 full S06 critic runs log_dir=$log_dir"
}

wait_for_upstream
validate_configs
run_smoke_stage
launch_full_stage
"""


if __name__ == "__main__":
    main()
