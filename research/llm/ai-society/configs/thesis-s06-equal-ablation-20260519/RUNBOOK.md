# Thesis S06 Equal Ablation 2026-05-19

S06-only thesis pilot with one comparison seed (`42`), three simulator-control levels, three tool-autonomy levels, and three 24-tick opportunity windows.

## Matrix

- 27 full runs: `proxy`, `scenario`, `pypsa` x `full`, `context_only`, `none` x Apr02/Apr09/Apr13.
- 9 smoke runs: one 2-tick Apr02 smoke for every simulator x tool-autonomy pair.
- Fixed society: `agent_count: 6`, `persona_profile: action_core_8`, `forecaster_backend: f8`.
- Fixed guard: `tool_policy: asset_simulator_v1`, `final_bid_guard: simulator_exact_match`, `safety_toolset: full`.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_thesis_s06_equal_ablation.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/thesis-s06-equal-ablation-20260519/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/thesis-s06-equal-ablation-20260519/all.txt
```

## Launch

Start both vLLM endpoints for `Qwen/Qwen3-32B`, then run:

```bash
cd /home/ucloud/heimdall
tmux new-session -d -s heimdall-tsa-s06 "bash ai-society/configs/thesis-s06-equal-ablation-20260519/run_thesis_s06_equal_ablation.sh > ai-society/configs/thesis-s06-equal-ablation-20260519/chain.log 2>&1"
```

## Monitor

```bash
tail -f ai-society/configs/thesis-s06-equal-ablation-20260519/chain.log
cat ai-society/configs/thesis-s06-equal-ablation-20260519/latest-smoke-log-dir.txt
cat ai-society/configs/thesis-s06-equal-ablation-20260519/latest-log-dir.txt
tail -f "$(cat ai-society/configs/thesis-s06-equal-ablation-20260519/latest-log-dir.txt)/sequential.stdout.log"
```

The stage runner validates simulator gating and preprobe provenance before continuing.
