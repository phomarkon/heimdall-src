# Verifierless Baseline 2026-05-19

This matrix launches after the current `thesis-s06-equal-ablation-20260519` TSA run completes cleanly.

## Matrix

- 15 full 24-tick runs.
- S06 action-core: `shadow-toolvisible` and `shadow-contextonly` on Apr02, Apr09, Apr13.
- Mixed20 side-aware: guarded real-controls, `shadow-toolvisible`, and `shadow-contextonly` on Apr02, Apr09, Apr13.
- Fixed seed and forecaster: `seed: 42`, `forecaster_backend: f8`.

## Safety Semantics

- `shadow-toolvisible`: simulator and feasibility tools are available, but final exact-match gating is disabled and every bid is shadow-scored after the decision.
- `shadow-contextonly`: pre-decision simulator, feasibility, candidate, ranker, and candidate-guidance tools are hidden. The prompt uses `unverified_bid_seeking`; submitted bids are shadow-scored only after the decision.
- S06 guarded comparators are the already-running TSA `scenario/full/medium` cells.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_verifierless_baseline.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/verifierless-baseline-20260519/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/verifierless-baseline-20260519/all.txt
```

## Compare

```bash
PYTHONPATH=. uv run python tools/evaluation/compare_verifierless_baseline.py
```
