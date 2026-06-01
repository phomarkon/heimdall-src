# Heimdall AI Society UCloud Runbook

This folder is laptop-safe by default. Editing, config validation, and dry-run
tests do not download LLM weights or start GPU services. Heavy work starts only
on the UCloud B200 server.

Current state: the society runner is implemented and supports both LLM and
deterministic-control runs. It writes `traces.jsonl` and `summary.json` under
`ai-society/runs/<run_id>/`; evaluation against ex-post activation truth is a
separate step under `tools/evaluation`.

## What To Run Where

Safe on laptop:

```bash
PYTHONPATH=. uv run python -m heimdall_ai_society validate-config ai-society/configs/local-dryrun.yaml
PYTHONPATH=. uv run pytest ai-society/tests -q
PYTHONPATH=. uv run python -m heimdall_ai_society run --config ai-society/configs/local-dryrun.yaml
```

Server-only native vLLM path:

```bash
bash setup.sh --cloud-ai-society
bash ai-society/ucloud-vllm/scripts/install_to_work.sh
cd /work/heimdall-vllm
bash scripts/launch_tmux.sh
source .venv/bin/activate
python scripts/healthcheck_vllm.py
python tests/test_heimdall_named_tool_call.py
source scripts/export_env.sh
cd /home/ucloud/heimdall
PYTHONPATH=. uv run python -m heimdall_ai_society run --config ai-society/configs/smoke-5.yaml
```

Docker Compose remains in `ai-society/docker-compose.yml` as an optional legacy
path for UCloud VMs where Docker works. Native Python + `uv` + vLLM is the
default path for UCloud interactive apps.

## Current Matrix Layout

Real AI-society experiment matrices should use the current dual-GPU layout:
two independent one-GPU vLLM servers, not tensor parallel by default. GPU0
serves `http://127.0.0.1:8000/v1`; GPU1 serves
`http://127.0.0.1:8001/v1`; each server keeps
`HEIMDALL_TENSOR_PARALLEL_SIZE=1` and `HEIMDALL_MAX_MODEL_LEN=16384`.

Prefer `ai-society/run_long_model_society_matrix.py` or the matching
`ai-society/runs/*/run_*.sh` wrapper for matrix work. Those launchers run
checked config lists sequentially, restart vLLM when the configured model
changes, and let each individual run use its configured LLM concurrency. Matrix
configs that call an LLM must include both local base URLs. For the current
Qwen3-32B balanced 12-agent setup, use `max_concurrency: 12` and
`per_endpoint_max_concurrency: 6`; smaller smoke, single-agent, or deterministic
controls can use lower concurrency intentionally.

Treat 8k serving as historical/model-ladder smoke territory only. Use 16k for
real matrices, including the next forecaster-zoo matrices. After every
successful matrix segment, evaluate the completed run directories, inspect the
small artefacts, commit, and push before starting the next segment.

## Fresh UCloud Checklist

1. Start an Ubuntu GPU server with a B200 allocation.
2. Confirm `/work` has enough persistent space for model weights.
3. Clone the branch:

```bash
git clone https://github.com/phomarkon/heimdall.git
cd heimdall
git checkout feat/ai-society-ucloud
```

4. Set Hugging Face auth only if needed:

```bash
export HF_TOKEN=...
```

5. Install repo prerequisites and validate GPU visibility:

```bash
HF_TOKEN=$HF_TOKEN bash setup.sh --cloud-ai-society
```

6. Install the native vLLM runtime into persistent UCloud storage:

```bash
bash ai-society/ucloud-vllm/scripts/install_to_work.sh
```

7. Start vLLM and verify the OpenAI-compatible endpoint:

```bash
cd /work/heimdall-vllm
bash scripts/launch_tmux.sh
tail -f logs/vllm.log
source .venv/bin/activate
python scripts/healthcheck_vllm.py
python tests/test_heimdall_named_tool_call.py
python tests/test_heimdall_n_agents.py --agents 5
source scripts/export_env.sh
```

8. Run the first AI society:

```bash
cd /home/ucloud/heimdall
PYTHONPATH=. uv run python -m heimdall_ai_society run --config ai-society/configs/smoke-5.yaml
```

Outputs land in `ai-society/runs/<run_id>/`:

- `traces.jsonl`: one line per agent decision per tick.
- `summary.json`: accepted/rejected/watched/abstained counts, model/endpoint metadata, forecast routing, memory audit, and side diagnostics.
- `cloud-runtime-manifest.json`: server/runtime metadata from setup.

Commit only small run artefacts. Do not add Hugging Face caches, virtualenvs,
model weights, raw data, local `.env` files, or vLLM logs.

## Native vLLM Defaults

Runtime files live in `/work/heimdall-vllm`. The standard Heimdall OpenAI
environment is:

```text
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=heimdall-local
HEIMDALL_LLM_MODEL=<model in /work/heimdall-vllm/.env>
```

The current proven default model is:

```text
HEIMDALL_MODEL=Qwen/Qwen3-32B
HEIMDALL_MAX_MODEL_LEN=16384
HEIMDALL_GPU_MEMORY_UTILIZATION=0.60
```

Use `HEIMDALL_MAX_MODEL_LEN=16384` for current AI-society experiment matrices.
The richer market/tool prompts plus `max_tokens=1000` can exceed an 8k served
context and cause vLLM context-limit `400 Bad Request` responses instead of real
generation. Treat 8k or smaller as smoke-test-only unless prompt size and output
budget are explicitly validated first.

For a lighter day-to-day dev default:

```text
HEIMDALL_MODEL=Qwen/Qwen3-4B
HEIMDALL_GPU_MEMORY_UTILIZATION=0.45
```

Known-good UCloud compatibility flags:

```text
HEIMDALL_ATTENTION_BACKEND=TRITON_ATTN
VLLM_USE_DEEP_GEMM=0
HEIMDALL_VLLM_EXTRA_ARGS="--enforce-eager --enable-auto-tool-choice --tool-call-parser qwen3_xml --moe-backend triton"
```

These flags avoid missing-`nvcc` FlashInfer/DeepGEMM JIT paths on the validated
UCloud image. They are compatibility-first; a CUDA-toolkit-capable image should
be tested later for maximum Blackwell throughput.

Tested public models:

```text
Qwen/Qwen3-0.6B
Qwen/Qwen3-1.7B
Qwen/Qwen3-4B
Qwen/Qwen3-8B
Qwen/Qwen3-14B
Qwen/Qwen3-30B-A3B
Qwen/Qwen3-32B
```

## Error Handling

`nvidia-smi: command not found`

- You are not on a GPU server image, or the NVIDIA driver is missing.
- Stop before vLLM. Fix the server image/driver first.

`/work does not exist`

- You are not on the expected UCloud environment.
- Do not install model/runtime state into the git clone.

vLLM fails with missing `nvcc` or `/usr/local/cuda`

- Keep `HEIMDALL_ATTENTION_BACKEND=TRITON_ATTN`.
- Keep `--enforce-eager`.
- For MoE models, keep `--moe-backend triton`.

Hugging Face model download failure

- Confirm the model id exists and is public, or export `HF_TOKEN`.
- Try a smaller public model temporarily:

```text
HEIMDALL_MODEL=Qwen/Qwen3-4B
```

vLLM out of memory

- Lower memory utilization or context first:

```text
HEIMDALL_GPU_MEMORY_UTILIZATION=0.45
HEIMDALL_MAX_MODEL_LEN=4096
```

vLLM healthcheck never passes

```bash
cd /work/heimdall-vllm
tail -200 logs/vllm.log
curl -v http://127.0.0.1:8000/v1/models
bash scripts/stop_vllm.sh
bash scripts/launch_tmux.sh
```

Invalid JSON/tool output from model

- The runner must treat LLM output as a proposal only.
- Validate every action with Pydantic before passing it to simulator/verifier.
- Prefer named `tool_choice` or `tool_choice="required"` for critical actions.

## Scaling Path

1. `local-dryrun.yaml`: validates code and trace writing without LLM.
2. Native vLLM `Qwen/Qwen3-4B`: quick UCloud smoke.
3. Native vLLM `Qwen/Qwen3-14B`: stronger society smoke.
4. Native vLLM `Qwen/Qwen3-32B`: current largest proven dense model.
5. Matrix runs: prefer checked config lists under `ai-society/configs/*/gpu0.txt`, `gpu1.txt`, or `config-list.txt` and the matching `run_*.py` launcher.
6. Larger societies: tune batching, context length, per-endpoint concurrency, memory utilization, and model size.

Initial local runs should use `f0` or `ar1` forecasters. Current real-context
matrix runs commonly use `f8`, with mixed side-aware profiles routing agents
across `f8`, `f7`, and `f3_ensemble`.

## Current Harness Semantics

Use these configs when restarting after a fresh UCloud allocation:

```bash
PYTHONPATH=. uv run python -m heimdall_ai_society validate-config ai-society/configs/p2h-local-dryrun.yaml
PYTHONPATH=. uv run python -m heimdall_ai_society run --config ai-society/configs/p2h-local-dryrun.yaml
PYTHONPATH=. uv run python -m heimdall_ai_society run --config ai-society/configs/p2h-ucloud-smoke-10.yaml
```

The harness separates three concepts:

- `get_activation_context` gives a non-leaking historical/context prior for whether an MTU is worth watching.
- `get_bid_feasibility` is a cheap heuristic evaluator for a candidate bid. It is advisory only.
- `simulate_bid` remains the P2H-only authoritative simulator/verifier check.
- `simulate_ev_bid`, `simulate_wind_bid`, `simulate_generator_bid`, `simulate_retailer_bid`, and `simulate_renewables_bid` are non-P2H simulator paths when enabled by `tool_policy` and `asset_simulator_mode`.
- `get_ev_bid_feasibility`, `get_wind_bid_feasibility`, `get_generator_bid_feasibility`, `get_retailer_bid_feasibility`, and `get_renewables_bid_feasibility` are advisory only; they must not be interpreted as authoritative verification.

Final bid rule: the runner accepts a final bid only when the trace contains an
accepted exact simulator call for the required archetype tool. P2H uses
`simulate_bid`; non-P2H action agents use their matching `simulate_*_bid` tool.
Context-only roles must `watch` or `abstain`, not bid.

Important config semantics:

- `chooser_mode: llm` calls the served model; `deterministic_best_accepted` and `deterministic_watch_threshold` are LLM-free controls and should set `llm.enabled: false`.
- `market_context: real` uses agent-visible cached context only, usually `data/cache/real_context/april_2026`.
- `tool_policy: p2h_only_simulator` keeps non-P2H agents advisory; `proxy_simulator` and `asset_simulator_v1` enable broader simulator-gated action.
- `asset_simulator_mode` supports three backend levels: `proxy`, `scenario_envelope`, and `pypsa_background`. Legacy `real` is an alias for `scenario_envelope`; dual-compare modes can keep proxy, scenario-envelope, or PyPSA-background as the controlling backend while recording comparison diagnostics. The PyPSA-background level applies the compact PyPSA-derived physical/network envelope and the mFRR clearing gate for every action archetype.
- `preprobe_mode` makes tool autonomy explicit. `full` preserves runner-probed synthesis; `context_only` seeds only non-bid context; `specialist_context` seeds context plus information-agent diagnostics while action agents must request candidate/feasibility/simulator tools; `none` starts without seeded tool records. New traces keep `tool_calls` backward-compatible and add per-call `provenance` plus aggregate provenance counters.
- `ablation_strategy` controls communication and action logic, including broadcast digest, risk filter, info-then-action, peer signal, retry council, central-supervisor, and society-chair variants. In `comm_central_supervisor`, specialists only report recommendations; a central supervisor chooses at most one exact accepted candidate and the deterministic execution gateway is the only path that records a market bid.
- `memory_enabled` injects lessons from `memory_bank_path`; `memory_scope_filter` controls all/archetype/agent/synthesis selection.

Cutoff rule: each persona sees data at `observed_at = tick_timestamp - info_latency_min`.
All context tools must filter to rows at or before `observed_at`; ex-post
activation truth remains available only to the evaluator after a run.

## Evaluation

Evaluate completed real-context runs after generation, never during prompting:

```bash
PYTHONPATH=. uv run python tools/evaluation/evaluate_society_run.py \
  --run-dir ai-society/runs/<run_id> \
  --context-dir data/cache/real_context/april_2026 \
  --truth-dir data/cache/evaluation_truth/april_2026
```

This writes `evaluations/<run_id>/bid_evaluations.parquet`,
`agent_metrics.parquet`, `archetype_metrics.parquet`, `run_summary.json`, and a
manifest with trace/context/truth hashes.

For paired LLM-vs-deterministic explanation checks:

```bash
PYTHONPATH=. uv run python tools/evaluation/evaluate_rationale_value.py \
  --output-dir evaluations/rationale-value
```

The rationale-value rubric is automatic and should be treated as an annotation
aid, not a replacement for human review.

## Saving Runs With Git

After `smoke-5` succeeds on UCloud, inspect the outputs:

```bash
ls -lah ai-society/runs/smoke-5
sed -n '1,80p' ai-society/runs/smoke-5/summary.json
sed -n '1,3p' ai-society/runs/smoke-5/traces.jsonl
```

Then commit only that run directory:

```bash
git status --short
git add ai-society/runs/smoke-5
git commit -m "exp: add smoke-5 ai society run"
git push origin feat/ai-society-ucloud
```

For later runs, use a unique `run_id` in a copied config so old results are not
overwritten:

```bash
cp ai-society/configs/smoke-10.yaml ai-society/configs/smoke-50.yaml
```
