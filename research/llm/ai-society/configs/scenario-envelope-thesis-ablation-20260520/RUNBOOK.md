# Scenario Envelope Thesis Ablation 2026-05-20

This matrix chains after `scenario-envelope-breadth-20260520` and fills the clean thesis comparison grid for s06, s12, and mixed20 societies.

## Matrix

- 18 breadth runs: s06-actioncore, s12-balanced, and s20-mixed x six April windows x seed 42.
- 12 seed robustness runs: s06-actioncore and s20-mixed x three core windows x seeds 13 and 137.
- 2 smoke runs on `apr02-0530`.
- All runs use `asset_simulator_mode: scenario_envelope`, `tool_policy: asset_simulator_v1`, `preprobe_mode: full`, and `candidate_sizing_mode: large`.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_scenario_envelope_thesis_ablation.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/scenario-envelope-thesis-ablation-20260520/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/scenario-envelope-thesis-ablation-20260520/all.txt
```

## Launch

```bash
cd /home/ucloud/heimdall
bash ai-society/configs/scenario-envelope-thesis-ablation-20260520/launch_after_scenario_envelope_breadth.sh
```
