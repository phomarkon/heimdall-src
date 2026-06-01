# Forecaster Zoo Priority Real-Control Screen

Config root:

```bash
ai-society/configs/forecaster-zoo-priority-realcontrol-20260518
```

## Validate

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
while read -r cfg; do
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1
done < ai-society/configs/forecaster-zoo-priority-realcontrol-20260518/all.txt
while read -r cfg; do
  uv run python -m heimdall_ai_society validate-config "$cfg" >/dev/null || exit 1
done < ai-society/configs/forecaster-zoo-priority-realcontrol-20260518/smoke.txt
```

## Smoke

Run the two 2-tick smoke configs before the full matrix:

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
while read -r cfg; do
  uv run python -m heimdall_ai_society run --config "$cfg" || exit 1
done < ai-society/configs/forecaster-zoo-priority-realcontrol-20260518/smoke.txt
```

## Full Matrix

Use the existing dual-GPU stage runner after both vLLM endpoints are up:

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
cfg_root=ai-society/configs/forecaster-zoo-priority-realcontrol-20260518
log_dir=ai-society/runs/forecaster-zoo-priority-realcontrol-20260518/logs-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$log_dir"

tmux new-session -d -s heimdall-fzpr "
cd /home/ucloud/heimdall &&
export PYTHONPATH=. &&
python ai-society/run_market_intelligence_stage.py \
  --stage forecaster-zoo-priority-realcontrol \
  --gpu gpu0 \
  --base-url http://127.0.0.1:8000/v1 \
  --config-list '$cfg_root/gpu0.txt' \
  --log-dir '$log_dir' \
  > '$log_dir/gpu0.stdout.log' 2>&1 &
python ai-society/run_market_intelligence_stage.py \
  --stage forecaster-zoo-priority-realcontrol \
  --gpu gpu1 \
  --base-url http://127.0.0.1:8001/v1 \
  --config-list '$cfg_root/gpu1.txt' \
  --log-dir '$log_dir' \
  > '$log_dir/gpu1.stdout.log' 2>&1 &
wait
"
```

## Evaluate Priority Calibration

After the matrix completes:

```bash
cd /home/ucloud/heimdall
export PYTHONPATH=.
uv run python tools/evaluation/evaluate_priority_calibration.py \
  --config-list ai-society/configs/forecaster-zoo-priority-realcontrol-20260518/all.txt \
  --output-dir evaluations/forecaster-zoo-priority-realcontrol-20260518
```
