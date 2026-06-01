#!/usr/bin/env bash
set -euo pipefail
cd /home/ucloud/heimdall
export PYTHONPATH=.

UPSTREAM_ROOT="ai-society/configs/sizing-forecaster-generalization-20260519"
UPSTREAM_SESSION="heimdall-sfg"
UPSTREAM_CHAIN_SESSION="heimdall-sfg-chain"
UPSTREAM_EXPECTED=9

ROOT="ai-society/configs/safety-context-only-unguarded-20260519"
RUN_ROOT="ai-society/runs/safety-context-only-unguarded-20260519"
TARGET_SESSION="heimdall-safetyctx"
TARGET_STAGE="safety-context-only-unguarded"
TARGET_EXPECTED=4

upstream_results_path() {
  if [ ! -f "$UPSTREAM_ROOT/latest-log-dir.txt" ]; then
    return 1
  fi
  local upstream_log_dir
  upstream_log_dir="$(cat "$UPSTREAM_ROOT/latest-log-dir.txt")"
  printf '%s/gpu0-results.json\n' "$upstream_log_dir"
}

wait_for_upstream() {
  while true; do
    status="missing-log-dir"
    if upstream_results="$(upstream_results_path)"; then
      status="missing-results"
      if [ -f "$upstream_results" ]; then
        status="$(UPSTREAM_RESULTS="$upstream_results" UPSTREAM_EXPECTED="$UPSTREAM_EXPECTED" uv run python - <<'PY_STATUS'
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
            echo "SFG matrix has failures; refusing to launch safety ablation." >&2
            cat "$upstream_results" >&2
            exit 1
            ;;
          complete)
            break
            ;;
        esac
      fi
    fi
    if ! tmux has-session -t "=$UPSTREAM_SESSION" 2>/dev/null && ! tmux has-session -t "=$UPSTREAM_CHAIN_SESSION" 2>/dev/null; then
      echo "$UPSTREAM_SESSION/$UPSTREAM_CHAIN_SESSION are gone but SFG does not show $UPSTREAM_EXPECTED successful rows; refusing to launch." >&2
      [ -n "${upstream_results:-}" ] && cat "$upstream_results" >&2 || true
      exit 1
    fi
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for SFG upstream ($status)"
    sleep 300
  done
}

refuse_completed_duplicates() {
  uv run python - <<'PY_DUPES'
from pathlib import Path
import json

root = Path("ai-society/configs/safety-context-only-unguarded-20260519")
run_root = Path("ai-society/runs/safety-context-only-unguarded-20260519")
completed = []
for line in (root / "all.txt").read_text().splitlines():
    cfg = Path(line.strip())
    if not cfg:
        continue
    run_id = cfg.stem
    summary = run_root / run_id / "summary.json"
    if summary.exists():
        try:
            payload = json.loads(summary.read_text())
            if payload.get("run_id") == run_id:
                completed.append(run_id)
            else:
                completed.append(run_id)
        except Exception:
            completed.append(run_id)
if completed:
    raise SystemExit("Completed duplicate run_id(s) already exist: " + ", ".join(completed))
PY_DUPES
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
    verify_smoke_trace "$cfg"
  done < "$ROOT/smoke.txt"
}

verify_smoke_trace() {
  local cfg="$1"
  CFG="$cfg" uv run python - <<'PY_SMOKE'
import json
import os
from pathlib import Path

from heimdall_ai_society.config import load_config

cfg = Path(os.environ["CFG"])
run_id = load_config(cfg).run_id or cfg.stem
run_dir = Path("ai-society/runs/safety-context-only-unguarded-20260519") / run_id
trace_path = run_dir / "traces.jsonl"
lines = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
hidden_prefixes = ("simulate_",)
hidden_names = {"simulate_bid", "candidate_menu", "rank_candidate_set"}
bad = []
bid_without_shadow = []
for row in lines:
    decision = row.get("decision") or {}
    calls = row.get("tool_calls") or []
    for call in calls:
        name = str(call.get("name") or "")
        if name == "shadow_required_simulation":
            continue
        if (
            name in hidden_names
            or name.startswith(hidden_prefixes)
            or (name.startswith("get_") and name.endswith("_bid_feasibility"))
        ):
            bad.append({"step": row.get("step"), "agent_id": row.get("agent_id"), "name": name})
    if decision.get("action") == "bid" and not any(call.get("name") == "shadow_required_simulation" for call in calls):
        bid_without_shadow.append({"step": row.get("step"), "agent_id": row.get("agent_id")})
if bad:
    raise SystemExit(f"pre-decision safety tools leaked: {bad[:5]}")
if bid_without_shadow:
    raise SystemExit(f"bids missing shadow simulation: {bid_without_shadow[:5]}")
print(f"verified smoke trace {run_id}: rows={len(lines)} bids={sum(1 for row in lines if (row.get('decision') or {}).get('action') == 'bid')}")
PY_SMOKE
}

matching_runner_active() {
  pgrep -af "run_market_intelligence_stage.py --stage $TARGET_STAGE .*--log-dir $RUN_ROOT/logs-" >/dev/null
}

other_stage_runner_active() {
  pgrep -af "run_market_intelligence_stage.py --stage " | grep -v -- "--stage $TARGET_STAGE " >/dev/null
}

wait_for_other_stage_runners() {
  while other_stage_runner_active; do
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for other active stage runner to clear"
    pgrep -af "run_market_intelligence_stage.py --stage " || true
    sleep 300
  done
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
  wait_for_other_stage_runners
  prepare_target_session
  log_dir="$RUN_ROOT/logs-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s\n' "$log_dir" > "$ROOT/latest-log-dir.txt"
  tmux new-session -d -s "$TARGET_SESSION" "
cd /home/ucloud/heimdall &&
export PYTHONPATH=. &&
uv run python ai-society/run_market_intelligence_stage.py \
  --stage '$TARGET_STAGE' \
  --gpu gpu0 \
  --base-url http://127.0.0.1:8000/v1 \
  --config-list '$ROOT/all.txt' \
  --log-dir '$log_dir' \
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
refuse_completed_duplicates
validate_configs
wait_for_other_stage_runners
run_smoke
launch_full
