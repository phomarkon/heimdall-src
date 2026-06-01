#!/usr/bin/env bash
# Freeform-matrix orchestrator: 6 cells x 3 windows x 2 seeds = 36 runs.
# Sequential. Resumable: skips configs whose run dir already has summary.json
# AND whose evaluations/<run_id>/ exists. Uses the 4 already-running vLLM
# endpoints on 8000-8003; does NOT restart vLLM.
set -uo pipefail
cd /home/ucloud/heimdall
export PYTHONPATH=".:ai-society/src${PYTHONPATH:+:${PYTHONPATH}}"

MATRIX_DIR="ai-society/runs/freeform-matrix-20260526"
CONFIG_LIST="ai-society/configs/freeform-matrix-20260526/config-list.txt"
LOG_DIR="${MATRIX_DIR}/logs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${LOG_DIR}"

CONTROLLER_LOG="${LOG_DIR}/controller.log"
RESULTS="${LOG_DIR}/results.jsonl"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${CONTROLLER_LOG}"; }

log "freeform-matrix start log_dir=${LOG_DIR}"

# Sanity: vLLM endpoints up?
for port in 8000 8001 8002 8003; do
    if ! curl -sf -H "Authorization: Bearer heimdall-local" "http://127.0.0.1:${port}/v1/models" >/dev/null; then
        log "ERROR: vLLM endpoint :${port} not responding — aborting"
        exit 1
    fi
done
log "vLLM endpoints :8000-:8003 healthy"

total=$(wc -l < "${CONFIG_LIST}")
idx=0
ok=0
skipped=0
failed=0

while IFS= read -r cfg; do
    [ -z "${cfg}" ] && continue
    idx=$((idx+1))
    run_id=$(basename "${cfg}" .yaml)
    run_dir="${MATRIX_DIR}/${run_id}"
    eval_dir="evaluations/${run_id}"

    if [ -f "${run_dir}/summary.json" ] && [ -d "${eval_dir}" ]; then
        log "[${idx}/${total}] SKIP ${run_id} (already complete)"
        skipped=$((skipped+1))
        continue
    fi

    log "[${idx}/${total}] RUN  ${run_id}"
    start=$(date +%s)
    if uv run python -m heimdall_ai_society run --config "${cfg}" \
            > "${LOG_DIR}/${run_id}.run.log" 2>&1; then
        run_elapsed=$(($(date +%s) - start))
        log "[${idx}/${total}] RUN  ok elapsed=${run_elapsed}s — evaluating"
        if uv run python tools/evaluation/evaluate_society_run.py \
                --run-dir "${run_dir}" \
                --context-dir data/cache/real_context/april_2026 \
                --truth-dir data/cache/evaluation_truth/april_2026 \
                --output-dir "${eval_dir}" \
                > "${LOG_DIR}/${run_id}.eval.log" 2>&1; then
            total_elapsed=$(($(date +%s) - start))
            log "[${idx}/${total}] EVAL ok total=${total_elapsed}s"
            printf '{"run_id":"%s","ok":true,"elapsed_seconds":%d}\n' "${run_id}" "${total_elapsed}" >> "${RESULTS}"
            ok=$((ok+1))
        else
            log "[${idx}/${total}] EVAL FAIL — see ${LOG_DIR}/${run_id}.eval.log"
            printf '{"run_id":"%s","ok":false,"stage":"eval"}\n' "${run_id}" >> "${RESULTS}"
            failed=$((failed+1))
        fi
    else
        log "[${idx}/${total}] RUN  FAIL — see ${LOG_DIR}/${run_id}.run.log"
        printf '{"run_id":"%s","ok":false,"stage":"run"}\n' "${run_id}" >> "${RESULTS}"
        failed=$((failed+1))
    fi
done < "${CONFIG_LIST}"

log "freeform-matrix done total=${idx} ok=${ok} skipped=${skipped} failed=${failed}"
