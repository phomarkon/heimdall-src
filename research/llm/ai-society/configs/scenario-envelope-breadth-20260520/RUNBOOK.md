# Scenario Envelope Breadth 2026-05-20

This matrix appends after `sim-backend-sizing-20260519` and broadens post-change scenario-envelope evidence.

## Matrix

- 4 smoke runs on `apr04-0600`.
- 16 scenario-envelope breadth runs: `s12-balanced` and `s20-mixed-persona` x medium/large x four rolling windows.
- 8 PyPSA tau comparison runs: `s12-balanced` x tau -50/-100 x four rolling windows.
- 2 full-day scenario-envelope examples on `apr17-0000` and `apr28-0000`.

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/experiments/generate_scenario_envelope_breadth.py --check-only
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/scenario-envelope-breadth-20260520/smoke.txt
while read -r cfg; do uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1; done < ai-society/configs/scenario-envelope-breadth-20260520/all.txt
```

## Launch

```bash
cd /home/ucloud/heimdall
bash ai-society/configs/scenario-envelope-breadth-20260520/launch_after_sbs.sh
```

## Compare

```bash
PYTHONPATH=. uv run python tools/evaluation/compare_scenario_envelope_breadth.py
```
