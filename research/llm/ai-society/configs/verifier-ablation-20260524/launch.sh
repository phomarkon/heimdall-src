#!/usr/bin/env bash
# Parallel launcher for the verifier-ablation matrix.
# Usage: bash launch.sh <list.txt> <concurrency>
set -uo pipefail
LIST="${1:?need a config list}"
CONC="${2:-3}"

run_one() {
  local cfg="$1"
  cd /home/ucloud/heimdall || return 1
  # cap math-lib threads per process: 384 cores / ~10 concurrent ~= 32 each, avoid thrash
  export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 NUMEXPR_NUM_THREADS=24 VECLIB_MAXIMUM_THREADS=24
  local logdir="/tmp/vab_logs"
  mkdir -p "$logdir"
  local rid; rid="$(basename "$cfg" .yaml)"
  local out="ai-society/runs/verifier-ablation-20260524/$rid"
  if [ -f "$out/summary.json" ]; then echo "SKIP $rid (exists)"; return 0; fi
  if PYTHONPATH=ai-society/src:. uv run python -m heimdall_ai_society run --config "$cfg" >"$logdir/$rid.log" 2>&1; then
    echo "OK   $rid"
  else
    echo "FAIL $rid -> $logdir/$rid.log"
  fi
}
export -f run_one

xargs -P "$CONC" -I {} bash -c 'run_one "$@"' _ {} < "$LIST"
echo "=== launch.sh done for $LIST ==="
