#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

PREFLIGHT_ONLY=0
if [[ "${1:-}" == "--preflight-only" ]]; then
  PREFLIGHT_ONLY=1
  shift
fi

LOG_DIR="ai-society/runs/profit-window-batch-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/batch.log"
STATUS_FILE="$LOG_DIR/status.json"
SUMMARY_COPY="$LOG_DIR/ablation-batch-summary.json"

CONFIGS=(
  ai-society/configs/profit-window-diverse-48/diverse-action-apr13-0015-48-q32.yaml
  ai-society/configs/profit-window-diverse-48/diverse-action-apr09-1830-48-q32.yaml
  ai-society/configs/profit-window-diverse-48/diverse-action-apr02-0530-48-q32.yaml
  ai-society/configs/profit-window-diverse-48/diverse-action-apr06-1300-48-q32.yaml
  ai-society/configs/profit-window-diverse-48/diverse-action-apr26-1400-48-q32.yaml
  ai-society/configs/profit-window-p2h-stress-24/p2h-stresstest-apr03-1915-24-q32.yaml
  ai-society/configs/profit-window-p2h-stress-24/p2h-stresstest-apr17-0745-24-q32.yaml
  ai-society/configs/profit-window-p2h-stress-24/p2h-stresstest-apr25-1600-24-q32.yaml
  ai-society/configs/profit-window-p2h-stress-24/p2h-stresstest-apr22-1430-24-q32.yaml
  ai-society/configs/profit-window-p2h-stress-24/p2h-stresstest-apr01-0415-24-q32.yaml
)

write_status() {
  local state="$1"
  local message="$2"
  python - "$STATUS_FILE" "$state" "$message" <<'PY'
import json
import sys
from datetime import UTC, datetime
path, state, message = sys.argv[1:4]
payload = {
    "updated_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
    "state": state,
    "message": message,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

{
  write_status "starting" "validating selected profitable-window configs"
  echo "[$(date -u +%FT%TZ)] Starting profit-window batch"
  echo "Log dir: $LOG_DIR"
  echo "Configs:"
  printf '  %s\n' "${CONFIGS[@]}"

  uv run python - <<'PY'
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

truth = pd.read_parquet("data/cache/evaluation_truth/april_2026/activation_truth.parquet")
truth = truth[truth["zone"] == "DK1"].sort_values("timestamp_utc").reset_index(drop=True)

def profit_per_mwh(row):
    if row["activation_direction"] == "up":
        return float(row["settlement_price_eur_mwh"] - row["spot_price_eur_mwh"])
    if row["activation_direction"] == "down":
        return float(row["spot_price_eur_mwh"] - row["settlement_price_eur_mwh"])
    return 0.0

truth["profit_per_mwh"] = truth.apply(profit_per_mwh, axis=1)
truth["oracle_eur"] = (truth["activated_volume_mwh"] * truth["profit_per_mwh"]).where(
    truth["profit_per_mwh"] > 0,
    0.0,
)

configs = [Path(p) for p in [
    "ai-society/configs/profit-window-diverse-48/diverse-action-apr13-0015-48-q32.yaml",
    "ai-society/configs/profit-window-diverse-48/diverse-action-apr09-1830-48-q32.yaml",
    "ai-society/configs/profit-window-diverse-48/diverse-action-apr02-0530-48-q32.yaml",
    "ai-society/configs/profit-window-diverse-48/diverse-action-apr06-1300-48-q32.yaml",
    "ai-society/configs/profit-window-diverse-48/diverse-action-apr26-1400-48-q32.yaml",
    "ai-society/configs/profit-window-p2h-stress-24/p2h-stresstest-apr03-1915-24-q32.yaml",
    "ai-society/configs/profit-window-p2h-stress-24/p2h-stresstest-apr17-0745-24-q32.yaml",
    "ai-society/configs/profit-window-p2h-stress-24/p2h-stresstest-apr25-1600-24-q32.yaml",
    "ai-society/configs/profit-window-p2h-stress-24/p2h-stresstest-apr22-1430-24-q32.yaml",
    "ai-society/configs/profit-window-p2h-stress-24/p2h-stresstest-apr01-0415-24-q32.yaml",
]]

days_by_family: dict[str, set[str]] = {"diverse": set(), "p2h": set()}
oracle_report = []
for config in configs:
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    validated = subprocess.run(
        ["uv", "run", "python", "-m", "heimdall_ai_society", "validate-config", str(config)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    payload = json.loads(validated.stdout)
    start = pd.Timestamp(raw["start_timestamp"])
    ticks = int(raw["ticks"])
    window = truth[(truth["timestamp_utc"] >= start) & (truth["timestamp_utc"] < start + pd.Timedelta(minutes=15 * ticks))]
    if len(window) != ticks:
        raise RuntimeError(f"{config} expected {ticks} truth rows, found {len(window)}")
    oracle = float(window["oracle_eur"].sum())
    if oracle <= 0:
        raise RuntimeError(f"{config} selected window has no oracle feasible profit")
    family = "diverse" if payload["ablation_strategy"] == "diverse_action_society" else "p2h"
    day = start.date().isoformat()
    if day in days_by_family[family]:
        raise RuntimeError(f"{family} repeats day {day}")
    days_by_family[family].add(day)
    oracle_report.append({
        "run_id": raw["run_id"],
        "start_timestamp": raw["start_timestamp"],
        "ticks": ticks,
        "oracle_screen_eur": round(oracle, 6),
        "positive_rows": int((window["oracle_eur"] > 0).sum()),
    })

print(json.dumps({"oracle_screen": oracle_report}, indent=2, sort_keys=True))
PY

  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    write_status "preflight_completed" "config validation and oracle-screen checks passed"
    echo "[$(date -u +%FT%TZ)] Preflight completed; not launching society runs"
    exit 0
  fi

  write_status "running" "batch runner active"
  uv run python ai-society/run_ablation_batch.py \
    --expected-model "Qwen/Qwen3-32B" \
    "${CONFIGS[@]}"

  cp ai-society/runs/ablation-batch-summary.json "$SUMMARY_COPY"
  write_status "completed" "all runs and evaluations completed"
  echo "[$(date -u +%FT%TZ)] Completed profit-window batch"
} 2>&1 | tee -a "$LOG_FILE"
