# Safety Context-Only Unguarded 2026-05-19

This matrix runs after `sizing-forecaster-generalization-20260519`.

## Purpose

Test whether LLM action agents submit bids that the current verifier/simulator loop would block when:

- pre-submit simulator and feasibility tools are hidden;
- final exact-match simulator acceptance is disabled;
- every submitted bid is shadow-scored after the decision.

## Cells

Four full 24-tick cells:

- `safetyctx-s06-actioncore-apr02-0530-seed13-q32`
- `safetyctx-s06-actioncore-apr02-0530-seed137-q32`
- `safetyctx-s20-mixed-apr09-1830-seed13-q32`
- `safetyctx-s20-mixed-apr09-1830-seed137-q32`

Guarded comparison runs already exist:

- `fco-s06-actioncore-bcast-apr02-0530-seed13-q32`
- `fco-s06-actioncore-bcast-apr02-0530-seed137-q32`
- `fco-s20-mixed-bcast-apr09-1830-seed13-q32`
- `fco-s20-mixed-bcast-apr09-1830-seed137-q32`

## Launch

```bash
cd /home/ucloud/heimdall
tmux new-session -d -s heimdall-safetyctx-chain \
  "cd /home/ucloud/heimdall && bash ai-society/configs/safety-context-only-unguarded-20260519/launch_after_sfg.sh > ai-society/configs/safety-context-only-unguarded-20260519/chain.log 2>&1"
```

## Monitor

```bash
tmux ls
tail -f ai-society/configs/safety-context-only-unguarded-20260519/chain.log
cat ai-society/configs/safety-context-only-unguarded-20260519/latest-log-dir.txt
tail -f "$(cat ai-society/configs/safety-context-only-unguarded-20260519/latest-log-dir.txt)/sequential-2gpu.stdout.log"
cat "$(cat ai-society/configs/safety-context-only-unguarded-20260519/latest-log-dir.txt)/gpu0-results.json"
```

## Trace Checks

Smoke and full runs should contain no pre-decision safety tools:

- no `simulate_*`
- no `get_*_bid_feasibility`
- no `candidate_menu`
- no `rank_candidate_set`

Submitted bids should contain a post-decision `shadow_required_simulation` tool record.

## Paired Comparison

After all four full runs complete and have been evaluated by the stage runner:

```bash
cd /home/ucloud/heimdall
PYTHONPATH=. uv run python tools/evaluation/compare_safety_context_ablation.py
```

This writes `evaluations/safety-context-only-unguarded-20260519/paired_summary.json`.
