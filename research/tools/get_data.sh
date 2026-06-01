#!/usr/bin/env bash
# get_data.sh — refetch the canonical DK1 panel from upstream sources.
#
# The processed DK1 panels (data/processed/*.parquet, ~6.5 MB) are committed to
# the repo for reproducibility. This script regenerates them from scratch by
# pulling raw ENTSO-E + Energinet history and rebuilding the train/val/test split.
#
# Required env (set in ~/.env_secrets, sourced from ~/.bashrc above the
# interactive guard):
#   ENTSOE_API_TOKEN  — Transparency Platform v2 token
#
# Usage:
#   bash tools/get_data.sh                    # incremental pull (skips existing months)
#   bash tools/get_data.sh --start 2025-01-01 # custom start date
#   bash tools/get_data.sh --fresh            # wipe data/raw + data/processed first
#
# Time: ~10 min for a full 2020-01-01 → 2026-04-30 fetch (rate-limited by ENTSO-E
# at 400 req/min). Idempotent — months already on disk are skipped.

set -euo pipefail
cd "$(dirname "$0")/.."

START="2020-01-01"
END="2026-04-30"
FRESH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2 ;;
    --end)   END="$2";   shift 2 ;;
    --fresh) FRESH=1;    shift   ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${ENTSOE_API_TOKEN:-}" ]]; then
  echo "ERROR: ENTSOE_API_TOKEN not set." >&2
  echo "       Add it to ~/.env_secrets and source ~/.bashrc, or export it now." >&2
  exit 1
fi

if [[ -d .venv ]]; then
  source .venv/bin/activate
elif command -v uv >/dev/null 2>&1; then
  PY="uv run --"
else
  echo "ERROR: no .venv and no uv available; install uv or run 'uv sync' first." >&2
  exit 1
fi
PY="${PY:-python}"

if [[ "$FRESH" == "1" ]]; then
  echo ">>> --fresh: wiping data/raw and data/processed"
  rm -rf data/raw data/processed
fi

mkdir -p data/raw data/processed

echo ">>> fetching DK1 history from ENTSO-E + Energinet ($START → $END)"
$PY tools/fetch_dk1_history.py --start "$START" --end "$END"

echo ">>> rebuilding canonical train/val/test panels (frozen pre/post-EAM split)"
$PY tools/build_dk1_panels.py --start "$START" --end "$END"

echo ">>> done. Panels written to data/processed/:"
ls -lh data/processed/dk1_panel_{train,val,test,full}.parquet 2>/dev/null || true
echo
echo "Verify split boundaries:"
$PY -c "
import polars as pl
for split in ('train','val','test'):
    df = pl.read_parquet(f'data/processed/dk1_panel_{split}.parquet')
    print(f'  {split}: rows={len(df):>7d}  ts={df[\"timestamp_utc\"].min()} → {df[\"timestamp_utc\"].max()}')
"
