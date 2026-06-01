# S06 Forecaster Breadth 2026-05-20

This matrix chains after `scenario-envelope-thesis-ablation-20260520` and isolates forecaster choice for the s06 action-core society.

## Matrix

- 36 full runs: 12 representative forecasters x 3 core windows.
- 3 smoke runs: f0, f8, and f10 on `apr02-0530`.
- All runs use `asset_simulator_mode: scenario_envelope`, `tool_policy: asset_simulator_v1`, `forecaster_routing_mode: run_level`, and `candidate_sizing_mode: large`.
- Fallbacks are resolved before launch and recorded in `manifest.json`; run configs contain the actual backend that will run.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_s06_forecaster_breadth.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/s06-forecaster-breadth-20260520/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/s06-forecaster-breadth-20260520/all.txt
```

## Launch

```bash
cd /home/ucloud/heimdall
bash ai-society/configs/s06-forecaster-breadth-20260520/launch_after_scenario_envelope_thesis_ablation.sh
```
