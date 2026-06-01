# Sizing Forecaster Generalization 2026-05-19

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
