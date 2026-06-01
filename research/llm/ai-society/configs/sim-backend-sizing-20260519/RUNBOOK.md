# Simulator Backend Sizing 2026-05-19

This guarded S06 matrix runs after the verifierless baseline matrix.

## Matrix

- 12 new full 24-tick S06 runs.
- Backends: `dual_compare_real_controls`, `dual_compare_pypsa_controls`.
- New sizing arms: `current`, `large`.
- Existing medium comparators are reused from the current TSA `scenario/full` and `pypsa/full` cells.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_sim_backend_sizing.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/sim-backend-sizing-20260519/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/sim-backend-sizing-20260519/all.txt
```

## Compare

```bash
PYTHONPATH=. uv run python tools/evaluation/compare_sim_backend_sizing.py
```
