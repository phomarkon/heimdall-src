from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

MATRIX = "central-supervisor-s06-20260522"
CONFIG_ROOT = Path("ai-society/configs") / MATRIX
RUN_ROOT = Path("ai-society/runs") / MATRIX
CONTEXT_DIR = "data/cache/real_context/april_2026"
MODEL = "Qwen/Qwen3-32B"

WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Generate {MATRIX} configs.")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    full, smoke = _payloads()
    _check(full, smoke)
    if args.check_only:
        print(json.dumps({"full": len(full), "smoke": len(smoke)}, sort_keys=True))
        return 0
    _write(full, smoke)
    return 0


def _payloads() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full = []
    smoke = []
    for window_slug, start in WINDOWS.items():
        full.append(_payload(window_slug, start, ticks=24, smoke=False))
        smoke.append(_payload(window_slug, start, ticks=2, smoke=True))
    return full, smoke


def _payload(window_slug: str, start: str, *, ticks: int, smoke: bool) -> dict[str, Any]:
    prefix = "smoke-css" if smoke else "css"
    run_id = f"{prefix}-s06-actioncore-central-supervisor-{window_slug}-seed42-{ticks}-q32"
    return {
        "run_id": run_id,
        "seed": 42,
        "forecaster_seed": 42,
        "zone": "DK1",
        "agent_count": 6,
        "ticks": ticks,
        "start_timestamp": start,
        "forecaster_backend": "f8",
        "forecaster_routing_mode": "persona",
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "market_context": "real",
        "context_dataset_dir": CONTEXT_DIR,
        "cache_refresh": False,
        "tool_mode": "openai_tools",
        "preprobe_mode": "full",
        "objective": "bid_seeking",
        "final_bid_guard": "simulator_exact_match",
        "safety_toolset": "full",
        "ablation_strategy": "comm_central_supervisor",
        "persona_profile": "action_core_8",
        "tool_policy": "asset_simulator_v1",
        "asset_simulator_mode": "scenario_envelope",
        "candidate_sizing_mode": "large",
        "candidate_sizing_max_candidates": 8,
        "candidate_sizing_min_mwh": 0.25,
        "candidate_sizing_cap_fraction": 1.0,
        "supervisor_soft_quota_per_24_ticks": 6,
        "supervisor_max_orders_per_tick": 1,
        "output_dir": str(RUN_ROOT),
        "llm": {
            "enabled": True,
            "model": MODEL,
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "api_key": "heimdall-local",
            "temperature": 0.2,
            "max_tokens": 1000,
            "timeout_seconds": 240,
            "max_concurrency": 12,
            "per_endpoint_max_concurrency": 6,
        },
    }


def _check(full: list[dict[str, Any]], smoke: list[dict[str, Any]]) -> None:
    if len(full) != 3:
        raise RuntimeError(f"expected 3 full configs, got {len(full)}")
    if len(smoke) != 3:
        raise RuntimeError(f"expected 3 smoke configs, got {len(smoke)}")
    run_ids = [payload["run_id"] for payload in [*full, *smoke]]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("duplicate run_id")
    for payload in [*full, *smoke]:
        if payload["ablation_strategy"] != "comm_central_supervisor":
            raise RuntimeError(f"bad strategy: {payload['run_id']}")
        if payload["supervisor_soft_quota_per_24_ticks"] != 6:
            raise RuntimeError(f"bad quota: {payload['run_id']}")
        if payload["llm"]["base_urls"] != ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"]:
            raise RuntimeError(f"bad endpoints: {payload['run_id']}")


def _write(full: list[dict[str, Any]], smoke: list[dict[str, Any]]) -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    full_paths = [_write_payload(payload, "full") for payload in full]
    smoke_paths = [_write_payload(payload, "smoke") for payload in smoke]
    (CONFIG_ROOT / "all.txt").write_text("\n".join(str(path) for path in full_paths) + "\n", encoding="utf-8")
    (CONFIG_ROOT / "smoke.txt").write_text("\n".join(str(path) for path in smoke_paths) + "\n", encoding="utf-8")
    (CONFIG_ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "matrix": MATRIX,
                "created_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
                "full_count": len(full_paths),
                "smoke_count": len(smoke_paths),
                "model": MODEL,
                "windows": WINDOWS,
                "supervisor_soft_quota_per_24_ticks": 6,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (CONFIG_ROOT / "RUNBOOK.md").write_text(_runbook(), encoding="utf-8")
    launcher = CONFIG_ROOT / "launch_after_current_matrix.sh"
    launcher.write_text(_launcher(), encoding="utf-8")
    launcher.chmod(0o755)


def _write_payload(payload: dict[str, Any], split: str) -> Path:
    path = CONFIG_ROOT / split / f"{payload['run_id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _runbook() -> str:
    return f"""# {MATRIX}

Central-supervisor S06 matrix over apr02-0530, apr09-1830, and apr13-0015.

Run:

```bash
bash {CONFIG_ROOT / "launch_after_current_matrix.sh"}
```
"""


def _launcher() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd /home/ucloud/heimdall
export PYTHONPATH=.

ROOT="{CONFIG_ROOT}"
RUN_ROOT="{RUN_ROOT}"
UPSTREAM_RUN_ROOT="ai-society/runs/deliberation-s12-and-s06-extended-large-20260522"
UPSTREAM_EXPECTED=6
TARGET_SESSION="heimdall-central-supervisor-s06"

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launch.lock"
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another central-supervisor launcher owns the lock; exiting"
  exit 0
fi

timestamp() {{ date -u +%Y-%m-%dT%H:%M:%SZ; }}

stage_status() {{
  local results="$1"
  local expected="$2"
  RESULTS="$results" EXPECTED="$expected" uv run python - <<'PY_STATUS'
import json
import os
from pathlib import Path
path = Path(os.environ["RESULTS"])
rows = json.loads(path.read_text()) if path.exists() else []
failed = [row for row in rows if row.get("ok") is False]
ok = [row for row in rows if row.get("ok") is True]
if failed:
    print("failed")
elif len(ok) >= int(os.environ["EXPECTED"]):
    print("complete")
else:
    print(f"running:{{len(ok)}}")
PY_STATUS
}}

latest_upstream_results() {{
  find "$UPSTREAM_RUN_ROOT"/logs* -name results.json -type f -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}}

wait_for_upstream() {{
  while true; do
    local results
    results="$(latest_upstream_results || true)"
    local status="missing-results"
    if [ -n "$results" ] && [ -f "$results" ]; then
      status="$(stage_status "$results" "$UPSTREAM_EXPECTED")"
      case "$status" in
        failed)
          echo "[$(timestamp)] upstream deliberation matrix failed; refusing central-supervisor launch." >&2
          cat "$results" >&2
          exit 1
          ;;
        complete)
          echo "[$(timestamp)] upstream deliberation matrix complete with $UPSTREAM_EXPECTED successful rows"
          break
          ;;
      esac
    fi
    if ! tmux has-session -t "=heimdall-delib-s12-s06-large-20260522" 2>/dev/null && ! pgrep -af "run_long_model_society_matrix.py.*deliberation-s12-and-s06-extended-large-20260522" >/dev/null; then
      echo "[$(timestamp)] upstream deliberation matrix is not running and did not complete cleanly ($status)." >&2
      [ -n "$results" ] && [ -f "$results" ] && cat "$results" >&2 || true
      exit 1
    fi
    echo "[$(timestamp)] waiting for upstream deliberation matrix ($status)"
    sleep 300
  done
}}

validate_configs() {{
  uv run python tools/experiments/generate_central_supervisor_s06_20260522.py --check-only >/dev/null
  while read -r cfg; do [ -z "$cfg" ] || uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null; done < "$ROOT/smoke.txt"
  while read -r cfg; do [ -z "$cfg" ] || uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null; done < "$ROOT/all.txt"
}}

run_smoke_stage() {{
  local log_dir="$RUN_ROOT/logs-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s\\n' "$log_dir" > "$ROOT/latest-smoke-log-dir.txt"
  echo "[$(timestamp)] smoke start central-supervisor-s06 log_dir=$log_dir"
  uv run python ai-society/run_long_model_society_matrix.py \\
    --config-list "$ROOT/smoke.txt" \\
    --log-dir "$log_dir" \\
    --continue-on-failure \\
    --skip-vllm-restart \\
    > "$log_dir/controller.stdout.log" 2>&1
  local status
  status="$(stage_status "$log_dir/results.json" 3)"
  if [ "$status" != "complete" ]; then
    echo "[$(timestamp)] smoke failed or incomplete: $status" >&2
    cat "$log_dir/results.json" >&2 || true
    exit 1
  fi
}}

launch_full_stage() {{
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
uv run python ai-society/run_long_model_society_matrix.py --config-list '$ROOT/all.txt' --log-dir '$log_dir' --continue-on-failure --skip-vllm-restart > '$log_dir/controller.stdout.log' 2>&1
"
  sleep 5
  if ! tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
    echo "[$(timestamp)] launch verification failed for $TARGET_SESSION" >&2
    tail -80 "$log_dir/controller.stdout.log" >&2 || true
    exit 1
  fi
  echo "[$(timestamp)] launched $TARGET_SESSION for 3 full central-supervisor S06 runs log_dir=$log_dir"
}}

wait_for_upstream
validate_configs
run_smoke_stage
launch_full_stage
"""


if __name__ == "__main__":
    raise SystemExit(main())
