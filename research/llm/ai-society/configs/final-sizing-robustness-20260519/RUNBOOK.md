# Final Sizing Robustness 2026-05-19

De-duplicated final-lock matrix queued after `large-bid-forecaster-breadth-20260518`.

## Cells

14 full cells:
- 8 core sizing cells: s06 and s20, Apr02/Apr09, medium/large sizing.
- 2 extra full-day large examples: s06 Apr09, s12 Apr03.
- 4 Apr02 large seed robustness cells: s06 and s20, seeds 13 and 137.

Two 2-tick smokes are listed in `smoke.txt`.

## Stack

All runs use Qwen/Qwen3-32B, real April context, priority calibration prompting, real-control asset simulation, and dual vLLM endpoints via `llm.base_urls`.

## Launch

`launch_after_lbfb.sh` waits for 21 successful upstream breadth rows, refuses upstream failures, refuses completed duplicate run_ids in this run root, validates configs, runs smokes, waits for other stage runners to clear, then starts detached tmux session `heimdall-final-sizing`.

Manual start of watcher:

```bash
cd /home/ucloud/heimdall
tmux new-session -d -s heimdall-final-sizing-chain "bash ai-society/configs/final-sizing-robustness-20260519/launch_after_lbfb.sh > ai-society/configs/final-sizing-robustness-20260519/chain.log 2>&1"
```

Monitor:

```bash
tmux ls
tail -f ai-society/configs/final-sizing-robustness-20260519/chain.log
cat ai-society/configs/final-sizing-robustness-20260519/latest-log-dir.txt
```
