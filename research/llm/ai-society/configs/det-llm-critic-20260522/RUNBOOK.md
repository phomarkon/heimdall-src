# Deterministic LLM Critic 2026-05-22

S06-only first pass for a forecast-diverse LLM critic. The deterministic proposer still selects exact simulator-backed candidates; Qwen3-32B can only keep the bid or veto it to watch/abstain.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_det_llm_critic_20260522.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/det-llm-critic-20260522/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/det-llm-critic-20260522/all.txt
```

## Launch

```bash
bash ai-society/configs/det-llm-critic-20260522/launch_after_current_matrix.sh
```
