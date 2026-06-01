# Large-Bid Forecaster Breadth Matrix

This matrix is intended to run after `forecaster-zoo-priority-realcontrol-20260518` completes.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
while read -r cfg; do
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1
done < ai-society/configs/large-bid-forecaster-breadth-20260518/all.txt
while read -r cfg; do
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1
done < ai-society/configs/large-bid-forecaster-breadth-20260518/smoke.txt
```

## Smoke

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
while read -r cfg; do
  uv run python -m heimdall_ai_society run --config "$cfg" || exit 1
done < ai-society/configs/large-bid-forecaster-breadth-20260518/smoke.txt
```

## Chained Launch

```bash
cd /home/ucloud/heimdall
bash ai-society/configs/large-bid-forecaster-breadth-20260518/launch_after_fzpr.sh
```

## Monitor

```bash
log_dir=$(cat ai-society/configs/large-bid-forecaster-breadth-20260518/latest-log-dir.txt)
tail -f "$log_dir/sequential-2gpu.stdout.log"
cat "$log_dir/gpu0-results.json"
```

## Priority Evaluation

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/evaluation/evaluate_priority_calibration.py \
  --config-list ai-society/configs/large-bid-forecaster-breadth-20260518/all.txt \
  --output-dir evaluations/large-bid-forecaster-breadth-20260518
```
