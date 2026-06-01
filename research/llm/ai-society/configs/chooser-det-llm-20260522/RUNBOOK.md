# Chooser Deterministic vs LLM 2026-05-22

This matrix compares whether Qwen3-32B adds value beyond greedy deterministic selection when simulator evidence is held constant.

## Matrix

- 45 full runs: deterministic, guarded LLM, and shadow-toolvisible LLM across three societies and five windows.
- 3 smoke runs: s06-actioncore on apr02-0530, one per chooser variant.
- All runs use `asset_simulator_mode: scenario_envelope`, `tool_policy: asset_simulator_v1`, `preprobe_mode: full`, `candidate_sizing_mode: medium`, `forecaster_backend: f8`, and seed 42.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_chooser_det_llm_20260522.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/chooser-det-llm-20260522/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/chooser-det-llm-20260522/all.txt
```

## Launch

```bash
cd /home/ucloud/heimdall
bash ai-society/configs/chooser-det-llm-20260522/launch_after_current_matrix.sh
```
