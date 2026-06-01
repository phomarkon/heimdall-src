#!/usr/bin/env bash
set -euo pipefail

cd /home/ucloud/heimdall
export PYTHONPATH=.

BASE_URL="http://127.0.0.1:8000/v1"

TSA_ROOT="ai-society/configs/thesis-s06-equal-ablation-20260519"
TSA_SESSION="heimdall-tsa-s06"
TSA_EXPECTED=27

VLB_ROOT="ai-society/configs/verifierless-baseline-20260519"
VLB_RUN_ROOT="ai-society/runs/verifierless-baseline-20260519"
VLB_SESSION="heimdall-vlb"
VLB_STAGE="verifierless-baseline"
VLB_EXPECTED=15

SBS_ROOT="ai-society/configs/sim-backend-sizing-20260519"
SBS_RUN_ROOT="ai-society/runs/sim-backend-sizing-20260519"
SBS_SESSION="heimdall-sbs"
SBS_STAGE="sim-backend-sizing"
SBS_EXPECTED=12

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

wait_for_tsa() {
  while true; do
    local status="missing-log-dir"
    local results=""
    if [ -f "$TSA_ROOT/latest-log-dir.txt" ]; then
      local log_dir
      log_dir="$(cat "$TSA_ROOT/latest-log-dir.txt")"
      results="$log_dir/gpu0-results.json"
      status="missing-results"
      if [ -f "$results" ]; then
        status="$(RESULTS="$results" EXPECTED="$TSA_EXPECTED" uv run python - <<'PY_STATUS'
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
)"
        case "$status" in
          failed)
            echo "[$(timestamp)] TSA matrix failed; refusing queued launch." >&2
            cat "$results" >&2
            exit 1
            ;;
          complete)
            echo "[$(timestamp)] TSA matrix complete with $TSA_EXPECTED successful rows"
            break
            ;;
        esac
      fi
    fi
    if ! tmux has-session -t "=$TSA_SESSION" 2>/dev/null; then
      echo "[$(timestamp)] $TSA_SESSION is gone but TSA does not show $TSA_EXPECTED successful rows; refusing queued launch." >&2
      [ -n "$results" ] && cat "$results" >&2 || true
      exit 1
    fi
    echo "[$(timestamp)] waiting for TSA upstream ($status)"
    sleep 300
  done
}

validate_family() {
  local root="$1"
  local generator="$2"
  uv run python "$generator" --check-only >/dev/null
  while read -r cfg; do
    [ -z "$cfg" ] && continue
    uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
  done < "$root/smoke.txt"
  while read -r cfg; do
    [ -z "$cfg" ] && continue
    uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null
  done < "$root/all.txt"
}

refuse_completed_duplicates() {
  local root="$1"
  local run_root="$2"
  uv run python - "$root" "$run_root" <<'PY_DUPES'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_root = Path(sys.argv[2])
completed = []
for list_name in ("smoke.txt", "all.txt"):
    for line in (root / list_name).read_text().splitlines():
        cfg = Path(line.strip())
        if not cfg:
            continue
        run_id = cfg.stem
        summary = run_root / run_id / "summary.json"
        evaluation = Path("evaluations") / run_id / "run_summary.json"
        for path in (summary, evaluation):
            if path.exists():
                try:
                    payload = json.loads(path.read_text())
                    if payload.get("run_id") in {None, run_id}:
                        completed.append(str(path))
                except Exception:
                    completed.append(str(path))
if completed:
    raise SystemExit("Completed duplicate run artifact(s) already exist:\n" + "\n".join(completed))
PY_DUPES
}

other_stage_runner_active() {
  local stage="$1"
  pgrep -af "run_market_intelligence_stage.py --stage " | grep -v -- "--stage $stage " >/dev/null
}

wait_for_other_stage_runners() {
  local stage="$1"
  while other_stage_runner_active "$stage"; do
    echo "[$(timestamp)] waiting for other active stage runner before $stage"
    pgrep -af "run_market_intelligence_stage.py --stage " || true
    sleep 300
  done
}

matching_runner_active() {
  local stage="$1"
  local run_root="$2"
  pgrep -af "run_market_intelligence_stage.py --stage $stage .*--log-dir $run_root/logs-" >/dev/null
}

run_smoke_stage() {
  local root="$1"
  local run_root="$2"
  local stage="$3"
  local log_dir="$run_root/logs-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s\n' "$log_dir" > "$root/latest-smoke-log-dir.txt"
  echo "[$(timestamp)] smoke start $stage log_dir=$log_dir"
  uv run python ai-society/run_market_intelligence_stage.py \
    --stage "$stage-smoke" \
    --gpu gpu0 \
    --base-url "$BASE_URL" \
    --config-list "$root/smoke.txt" \
    --log-dir "$log_dir" \
    > "$log_dir/sequential.stdout.log" 2>&1
  echo "[$(timestamp)] smoke complete $stage"
}

prepare_target_session() {
  local session="$1"
  local stage="$2"
  local run_root="$3"
  if tmux has-session -t "=$session" 2>/dev/null; then
    if matching_runner_active "$stage" "$run_root"; then
      echo "tmux session $session already runs $stage; not launching duplicate."
      return 1
    fi
    echo "tmux session $session exists but no matching runner is active; removing stale session."
    tmux kill-session -t "=$session"
  fi
  return 0
}

launch_full_stage() {
  local root="$1"
  local run_root="$2"
  local session="$3"
  local stage="$4"
  local expected="$5"

  wait_for_other_stage_runners "$stage"
  prepare_target_session "$session" "$stage" "$run_root" || return 0

  local log_dir="$run_root/logs-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$log_dir"
  printf '%s\n' "$log_dir" > "$root/latest-log-dir.txt"
  tmux new-session -d -s "$session" "
cd /home/ucloud/heimdall &&
export PYTHONPATH=. &&
uv run python ai-society/run_market_intelligence_stage.py \
  --stage '$stage' \
  --gpu gpu0 \
  --base-url '$BASE_URL' \
  --config-list '$root/all.txt' \
  --log-dir '$log_dir' \
  > '$log_dir/sequential.stdout.log' 2>&1
"
  verify_launch "$session" "$stage" "$run_root" "$log_dir" "$expected"
  wait_for_stage_results "$session" "$log_dir/gpu0-results.json" "$expected"
}

verify_launch() {
  local session="$1"
  local stage="$2"
  local run_root="$3"
  local log_dir="$4"
  local expected="$5"
  for _ in $(seq 1 12); do
    if tmux has-session -t "=$session" 2>/dev/null && matching_runner_active "$stage" "$run_root" && grep -q "1/$expected start" "$log_dir/sequential.stdout.log" 2>/dev/null; then
      echo "[$(timestamp)] verified $session started $stage"
      return 0
    fi
    sleep 5
  done
  echo "Launch verification failed for $session." >&2
  tmux has-session -t "=$session" 2>/dev/null || echo "missing tmux session" >&2
  matching_runner_active "$stage" "$run_root" || echo "missing matching runner process" >&2
  tail -80 "$log_dir/sequential.stdout.log" >&2 || true
  exit 1
}

wait_for_stage_results() {
  local session="$1"
  local results="$2"
  local expected="$3"
  while true; do
    local status="missing-results"
    if [ -f "$results" ]; then
      status="$(RESULTS="$results" EXPECTED="$expected" uv run python - <<'PY_STATUS'
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
)"
      case "$status" in
        failed)
          echo "[$(timestamp)] stage failed; refusing to continue." >&2
          cat "$results" >&2
          exit 1
          ;;
        complete)
          echo "[$(timestamp)] stage complete with $expected successful rows"
          return 0
          ;;
      esac
    fi
    if ! tmux has-session -t "=$session" 2>/dev/null; then
      echo "[$(timestamp)] $session is gone but $results does not show $expected successful rows; refusing to continue." >&2
      [ -f "$results" ] && cat "$results" >&2 || true
      exit 1
    fi
    echo "[$(timestamp)] waiting for $session ($status)"
    sleep 300
  done
}

wait_for_tsa

echo "[$(timestamp)] validating queued matrices"
validate_family "$VLB_ROOT" "tools/experiments/generate_verifierless_baseline.py"
validate_family "$SBS_ROOT" "tools/experiments/generate_sim_backend_sizing.py"
refuse_completed_duplicates "$VLB_ROOT" "$VLB_RUN_ROOT"
refuse_completed_duplicates "$SBS_ROOT" "$SBS_RUN_ROOT"

run_smoke_stage "$VLB_ROOT" "$VLB_RUN_ROOT" "$VLB_STAGE"
launch_full_stage "$VLB_ROOT" "$VLB_RUN_ROOT" "$VLB_SESSION" "$VLB_STAGE" "$VLB_EXPECTED"
PYTHONPATH=. uv run python tools/evaluation/compare_verifierless_baseline.py

run_smoke_stage "$SBS_ROOT" "$SBS_RUN_ROOT" "$SBS_STAGE"
launch_full_stage "$SBS_ROOT" "$SBS_RUN_ROOT" "$SBS_SESSION" "$SBS_STAGE" "$SBS_EXPECTED"
PYTHONPATH=. uv run python tools/evaluation/compare_sim_backend_sizing.py

bash ai-society/configs/scenario-envelope-breadth-20260520/launch_after_sbs.sh

echo "[$(timestamp)] queued verifierless, simulator-sizing, and scenario-envelope-breadth matrices complete"
