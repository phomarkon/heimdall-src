# Regular-Bid Breadth + Proxy Gap Matrix

This matrix runs after `large-bid-forecaster-breadth-20260518` completes. It fills regular-bid 24-tick results across the same broader windows, with `s06`, `s12`, and `s20-mixed` main configurations plus a focused proxy-control subset.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
while read -r cfg; do
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1
done < ai-society/configs/regular-bid-breadth-proxy-gap-20260518/all.txt
while read -r cfg; do
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1
done < ai-society/configs/regular-bid-breadth-proxy-gap-20260518/smoke.txt
```

## Chained Launch

```bash
cd /home/ucloud/heimdall
bash ai-society/configs/regular-bid-breadth-proxy-gap-20260518/launch_after_lbfb.sh
```

## Monitor

```bash
log_dir=$(cat ai-society/configs/regular-bid-breadth-proxy-gap-20260518/latest-log-dir.txt)
tail -f "$log_dir/sequential-2gpu.stdout.log"
cat "$log_dir/gpu0-results.json"
```
