# Queued Verifierless And Simulator-Sizing 2026-05-19

This chain waits for `heimdall-tsa-s06` / `thesis-s06-equal-ablation-20260519` to finish with 27 successful full rows, then launches:

1. `verifierless-baseline-20260519`
2. `sim-backend-sizing-20260519`

Both matrices use F8, seed 42, DK1 real April context, and the three 24-tick windows `apr02-0530`, `apr09-1830`, and `apr13-0015`.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_verifierless_baseline.py --check-only
uv run python tools/experiments/generate_sim_backend_sizing.py --check-only
```

## Launch

```bash
cd /home/ucloud/heimdall
tmux new-session -d -s heimdall-vlb-sbs-chain \
  "cd /home/ucloud/heimdall && bash ai-society/configs/queued-verifierless-and-sizing-20260519/launch_after_tsa.sh > ai-society/configs/queued-verifierless-and-sizing-20260519/chain.log 2>&1"
```

## Monitor

```bash
tail -f ai-society/configs/queued-verifierless-and-sizing-20260519/chain.log

cat ai-society/configs/verifierless-baseline-20260519/latest-log-dir.txt
tail -f "$(cat ai-society/configs/verifierless-baseline-20260519/latest-log-dir.txt)/sequential.stdout.log"
cat "$(cat ai-society/configs/verifierless-baseline-20260519/latest-log-dir.txt)/gpu0-results.json"

cat ai-society/configs/sim-backend-sizing-20260519/latest-log-dir.txt
tail -f "$(cat ai-society/configs/sim-backend-sizing-20260519/latest-log-dir.txt)/sequential.stdout.log"
cat "$(cat ai-society/configs/sim-backend-sizing-20260519/latest-log-dir.txt)/gpu0-results.json"
```

## Outputs

- `evaluations/verifierless-baseline-20260519/paired_summary.json`
- `evaluations/sim-backend-sizing-20260519/summary.json`
