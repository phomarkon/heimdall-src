#!/bin/bash
# Overnight orchestrator (2026-05-25): wait for the NOW d3-breadth batch, fail-fast smoke the new
# matrices, then launch the 264-run overnight sharded across all 4 GPUs. Quota-safe (cached Qwen3-32B).
set -u
cd /home/ucloud/heimdall
export PYTHONPATH=.:ai-society/src OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
LOG=logs/overnight-20260525; mkdir -p "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG/orchestrator.log"; }
MHS=ai-society/configs/matched-hetero-seedext-20260525
SZX=ai-society/configs/sizing-seedext-20260525

say "waiting for NOW d3-breadth batch to finish..."
while pgrep -f "run_ablation_batch.py.*d3-breadth-hetero" >/dev/null; do sleep 60; done
say "NOW batch clear. running fail-fast smokes."

cat "$MHS/smoke.txt" "$SZX/smoke.txt" > "$LOG/smoke.txt"
mapfile -t S < "$LOG/smoke.txt"
uv run python ai-society/run_ablation_batch.py "${S[@]}" --expected-model Qwen/Qwen3-32B \
  --base-url http://127.0.0.1:8000/v1 --continue-on-failure > "$LOG/smoke.log" 2>&1

# gate: every smoke run must have produced a non-empty traces.jsonl
NSM=${#S[@]}
OK=$(find ai-society/runs/matched-hetero-seedext-20260525 ai-society/runs/sizing-seedext-20260525 \
     -name traces.jsonl 2>/dev/null | xargs -r wc -l 2>/dev/null | awk '$1>0&&$2!="total"{c++}END{print c+0}')
say "smoke produced $OK/$NSM non-empty trace files"
if [ "$OK" -lt "$NSM" ]; then say "SMOKE GATE FAILED — NOT launching overnight. Inspect $LOG/smoke.log"; exit 1; fi
say "smoke gate passed. launching 264-run overnight, 4 shards."

# combine all overnight configs, interleave (round-robin) so each shard mixes light/heavy societies
cat "$MHS/cdl_seedext.txt" "$MHS/hetero_bc.txt" "$SZX/full.txt" \
    ai-society/configs/d3-breadth-hetero-20260525/overnight.txt > "$LOG/all.txt"
awk 'NF' "$LOG/all.txt" | awk '{print > "'"$LOG"'/shard_" (NR%4) ".txt"}'
for s in 0 1 2 3; do
  mapfile -t C < "$LOG/shard_$s.txt"
  nohup uv run python ai-society/run_ablation_batch.py "${C[@]}" --expected-model Qwen/Qwen3-32B \
    --base-url http://127.0.0.1:8000/v1 --continue-on-failure > "$LOG/shard_$s.log" 2>&1 &
  say "  shard $s -> pid $! (${#C[@]} configs)"
done
say "all 4 overnight shards launched. total $(awk 'NF' "$LOG/all.txt" | wc -l) runs."
wait
say "OVERNIGHT COMPLETE."
