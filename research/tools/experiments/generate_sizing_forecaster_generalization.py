from __future__ import annotations

from pathlib import Path

ROOT = Path("ai-society/configs/sizing-forecaster-generalization-20260519")
RUN_ROOT = "ai-society/runs/sizing-forecaster-generalization-20260519"

WINDOWS = [
    ("apr03-1430", "2026-04-03T14:30:00Z"),
    ("apr05-1030", "2026-04-05T10:30:00Z"),
    ("apr06-1300", "2026-04-06T13:00:00Z"),
    ("apr07-1715", "2026-04-07T17:15:00Z"),
    ("apr22-0830", "2026-04-22T08:30:00Z"),
    ("apr27-1730", "2026-04-27T17:30:00Z"),
]

SETUPS = {
    "s06-actioncore": {
        "agent_count": 6,
        "forecaster_backend": "f9",
        "forecaster_routing_mode": "run_level",
        "persona_profile": "action_core_8",
    },
    "s12-balanced": {
        "agent_count": 12,
        "forecaster_backend": "f9",
        "forecaster_routing_mode": "run_level",
        "persona_profile": "balanced_intelligence",
    },
    "s20-mixed-persona": {
        "agent_count": 20,
        "forecaster_backend": "f8",
        "forecaster_routing_mode": "persona",
        "persona_profile": "mixed_expert_20_sideaware",
    },
}


def config_text(run_id: str, setup: dict[str, object], timestamp: str, ticks: int, sizing: str) -> str:
    return f"""run_id: {run_id}
seed: 42
forecaster_seed: 42
zone: DK1
agent_count: {setup["agent_count"]}
ticks: {ticks}
start_timestamp: '{timestamp}'
forecaster_backend: {setup["forecaster_backend"]}
forecaster_routing_mode: {setup["forecaster_routing_mode"]}
chooser_mode: llm
verifier_mode: simulator
verifier_tau_eur: 0.0
market_context: real
tool_mode: openai_tools
objective: bid_seeking
ablation_strategy: comm_broadcast_digest_priority_calibration
persona_profile: {setup["persona_profile"]}
scenario_id: p2h_dk1_pypsa
tool_policy: asset_simulator_v1
asset_simulator_mode: dual_compare_real_controls
asset_proxy_style: market
candidate_sizing_mode: {sizing}
candidate_sizing_cap_fraction: 1.0
candidate_sizing_min_mwh: 0.25
candidate_sizing_max_candidates: 8
max_tool_rounds: 6
simulator_max_concurrency: 8
data_start: '2026-04-01T00:00:00Z'
data_end: '2026-05-01T00:00:00Z'
context_dataset_dir: data/cache/real_context/april_2026
data_cache_dir: data/cache/real_context/april_2026/source_cache
default_lookback_hours: 24
cache_refresh: false
output_dir: {RUN_ROOT}
memory_enabled: true
memory_bank_path: ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl
memory_max_items_per_agent: 5
memory_max_prompt_chars: 2400
reviewer_mode: code_only
llm:
  enabled: true
  base_urls:
  - http://127.0.0.1:8000/v1
  - http://127.0.0.1:8001/v1
  api_key: heimdall-local
  model: Qwen/Qwen3-32B
  temperature: 0.2
  max_tokens: 640
  timeout_seconds: 180
  max_concurrency: 12
  per_endpoint_max_concurrency: 6
"""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    all_paths: list[str] = []
    smoke_paths: list[str] = []

    for window, timestamp in WINDOWS:
        for setup_name, setup in SETUPS.items():
            run_id = f"sfg-{setup_name}-{window}-24-medium-real-q32"
            path = ROOT / f"{run_id}.yaml"
            write(path, config_text(run_id, setup, timestamp, 24, "medium"))
            all_paths.append(str(path))

    for window, timestamp in [
        ("apr05-1030", "2026-04-05T10:30:00Z"),
        ("apr06-1300", "2026-04-06T13:00:00Z"),
    ]:
        for setup_name in ("s06-actioncore", "s12-balanced"):
            setup = SETUPS[setup_name]
            run_id = f"sfg-{setup_name}-{window}-24-large-f9-real-q32"
            path = ROOT / f"{run_id}.yaml"
            write(path, config_text(run_id, setup, timestamp, 24, "large"))
            all_paths.append(str(path))

    for setup_name, day, timestamp in [
        ("s12-balanced", "apr07-0000", "2026-04-07T00:00:00Z"),
        ("s20-mixed-persona", "apr09-0000", "2026-04-09T00:00:00Z"),
    ]:
        setup = SETUPS[setup_name]
        run_id = f"sfg-{setup_name}-{day}-96-medium-real-q32"
        path = ROOT / f"{run_id}.yaml"
        write(path, config_text(run_id, setup, timestamp, 96, "medium"))
        all_paths.append(str(path))

    for setup_name, sizing in [("s12-balanced", "medium"), ("s20-mixed-persona", "medium")]:
        setup = SETUPS[setup_name]
        run_id = f"smoke-sfg-{setup_name}-apr03-1430-2-{sizing}-real-q32"
        path = ROOT / f"{run_id}.yaml"
        write(path, config_text(run_id, setup, "2026-04-03T14:30:00Z", 2, sizing))
        smoke_paths.append(str(path))

    write(ROOT / "all.txt", "\n".join(all_paths) + "\n")
    write(ROOT / "smoke.txt", "\n".join(smoke_paths) + "\n")
    write(ROOT / "RUNBOOK.md", RUNBOOK)
    write(ROOT / "launch_after_rbbpg.sh", LAUNCHER)


RUNBOOK = """# Sizing Forecaster Generalization 2026-05-19

This matrix chains after `regular-bid-breadth-proxy-gap-20260518`. It tests whether medium/large bid sizing generalizes across the broad windows, with f9 forecaster arms for conservative s06/s12 and f8 persona routing for s20.

## Cells

- 18 medium 24-tick cells across s06/s12/s20 and 6 broad windows.
- 4 large f9 contrast cells for s06/s12 on Apr05 and Apr06.
- 2 medium 96-tick cells for longer-horizon robustness.

## Launch

```bash
cd /home/ucloud/heimdall
bash ai-society/configs/sizing-forecaster-generalization-20260519/launch_after_rbbpg.sh
```

## Monitor

```bash
log_dir=$(cat ai-society/configs/sizing-forecaster-generalization-20260519/latest-log-dir.txt)
tail -f "$log_dir/sequential-2gpu.stdout.log"
cat "$log_dir/gpu0-results.json"
```
"""

LAUNCHER = """#!/usr/bin/env bash
set -euo pipefail
cd /home/ucloud/heimdall
export PYTHONPATH=.

UPSTREAM_ROOT="ai-society/configs/regular-bid-breadth-proxy-gap-20260518"
UPSTREAM_LOG_DIR="$(cat "$UPSTREAM_ROOT/latest-log-dir.txt")"
UPSTREAM_RESULTS="$UPSTREAM_LOG_DIR/gpu0-results.json"
UPSTREAM_SESSION="heimdall-rbbpg"
UPSTREAM_EXPECTED=24

ROOT="ai-society/configs/sizing-forecaster-generalization-20260519"
RUN_ROOT="ai-society/runs/sizing-forecaster-generalization-20260519"
TARGET_SESSION="heimdall-sfg"
TARGET_STAGE="sizing-forecaster-generalization"
TARGET_EXPECTED=24

wait_for_upstream() {
  while true; do
    status="missing"
    if [ -f "$UPSTREAM_RESULTS" ]; then
      status="$(UPSTREAM_RESULTS="$UPSTREAM_RESULTS" UPSTREAM_EXPECTED="$UPSTREAM_EXPECTED" uv run python - <<'PY_STATUS'
import json
import os
from pathlib import Path

path = Path(os.environ["UPSTREAM_RESULTS"])
expected = int(os.environ["UPSTREAM_EXPECTED"])
rows = json.loads(path.read_text()) if path.exists() else []
failed = [r for r in rows if r.get("ok") is False]
ok = [r for r in rows if r.get("ok") is True]
if failed:
    print("failed")
elif len(ok) >= expected:
    print("complete")
else:
    print(f"running:{len(ok)}")
PY_STATUS
)"
      case "$status" in
        failed)
          echo "Upstream matrix has failures; refusing to launch next matrix." >&2
          cat "$UPSTREAM_RESULTS" >&2
          exit 1
          ;;
        complete)
          break
          ;;
      esac
    fi
    if ! tmux has-session -t "=$UPSTREAM_SESSION" 2>/dev/null; then
      echo "$UPSTREAM_SESSION is gone but upstream matrix does not show $UPSTREAM_EXPECTED successful rows; refusing to launch." >&2
      cat "$UPSTREAM_RESULTS" >&2 || true
      exit 1
    fi
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for upstream matrix ($status)"
    sleep 300
  done
}

validate_configs() {
  while read -r cfg; do
    uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1
  done < "$ROOT/all.txt"
  while read -r cfg; do
    uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1
  done < "$ROOT/smoke.txt"
}

run_smoke() {
  while read -r cfg; do
    echo "[smoke] $cfg"
    uv run python -m heimdall_ai_society run --config "$cfg" || exit 1
  done < "$ROOT/smoke.txt"
}

matching_runner_active() {
  pgrep -af "run_market_intelligence_stage.py --stage $TARGET_STAGE .*--log-dir $RUN_ROOT/logs-" >/dev/null
}

prepare_target_session() {
  if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
    if matching_runner_active; then
      echo "tmux session $TARGET_SESSION already runs $TARGET_STAGE; not launching duplicate."
      exit 0
    fi
    echo "tmux session $TARGET_SESSION exists but no matching runner is active; removing stale session."
    tmux kill-session -t "=$TARGET_SESSION"
  fi
}

launch_full() {
  prepare_target_session
  log_dir="$RUN_ROOT/logs-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s\n' "$log_dir" > "$ROOT/latest-log-dir.txt"
  tmux new-session -d -s "$TARGET_SESSION" "
cd /home/ucloud/heimdall &&
export PYTHONPATH=. &&
uv run python ai-society/run_market_intelligence_stage.py \\
  --stage '$TARGET_STAGE' \\
  --gpu gpu0 \\
  --base-url http://127.0.0.1:8000/v1 \\
  --config-list '$ROOT/all.txt' \\
  --log-dir '$log_dir' \\
  > '$log_dir/sequential-2gpu.stdout.log' 2>&1
"
  verify_launch "$log_dir"
  echo "$log_dir"
}

verify_launch() {
  log_dir="$1"
  for _ in $(seq 1 12); do
    if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null && matching_runner_active && grep -q "1/$TARGET_EXPECTED start" "$log_dir/sequential-2gpu.stdout.log" 2>/dev/null; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] verified $TARGET_SESSION started $TARGET_STAGE"
      return 0
    fi
    sleep 5
  done
  echo "Launch verification failed for $TARGET_SESSION." >&2
  tmux has-session -t "=$TARGET_SESSION" 2>/dev/null || echo "missing tmux session" >&2
  matching_runner_active || echo "missing matching runner process" >&2
  tail -80 "$log_dir/sequential-2gpu.stdout.log" >&2 || true
  exit 1
}

wait_for_upstream
validate_configs
run_smoke
launch_full
"""


if __name__ == "__main__":
    main()
