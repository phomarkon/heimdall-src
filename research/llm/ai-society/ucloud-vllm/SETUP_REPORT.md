# Heimdall Native vLLM UCloud Reference Report

This is the sanitized reference report for the native UCloud vLLM setup used by
the Heimdall AI society branch. It intentionally excludes live `.env` files,
virtualenvs, Hugging Face caches, model weights, and full logs.

## Environment Provenance

Verified on a UCloud Ubuntu GPU server:

```text
OS: Ubuntu 24.04.4 LTS
GPU: NVIDIA B200, 183359 MiB, compute capability 10.0
Driver: 595.71.05
nvidia-smi CUDA: 13.2
Python: 3.12.13 via uv managed Python
torch: 2.11.0+cu130
torch CUDA: 13.0
vLLM: 0.20.2
```

Docker/Docker Compose was not available in the UCloud interactive environment
used for validation, so the working path is native Python + `uv` + vLLM under
`/work/heimdall-vllm`.

## Install And Launch

From a fresh clone on UCloud:

```bash
cd heimdall
bash setup.sh --cloud-ai-society
bash ai-society/ucloud-vllm/scripts/install_to_work.sh
cd /work/heimdall-vllm
bash scripts/launch_tmux.sh
tail -f logs/vllm.log
```

Then verify:

```bash
source .venv/bin/activate
python scripts/healthcheck_vllm.py
python tests/test_heimdall_named_tool_call.py
python tests/test_heimdall_n_agents.py --agents 5
```

Export the Heimdall-compatible OpenAI environment:

```bash
source scripts/export_env.sh
```

This emits:

```text
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=heimdall-local
HEIMDALL_LLM_MODEL=<model from .env>
```

## Working UCloud Flags

The UCloud image validated here did not expose `nvcc` or `/usr/local/cuda`.
Default vLLM kernel choices attempted FlashInfer/DeepGEMM JIT paths and failed.
The known-good compatibility flags are:

```text
HEIMDALL_ATTENTION_BACKEND=TRITON_ATTN
VLLM_USE_DEEP_GEMM=0
HEIMDALL_VLLM_EXTRA_ARGS="--enforce-eager --enable-auto-tool-choice --tool-call-parser qwen3_xml --moe-backend triton"
```

Effects:

- `TRITON_ATTN` avoids FlashInfer attention JIT requiring `nvcc`.
- `--enforce-eager` disables CUDA graphs and torch compile for compatibility.
- `qwen3_xml` enables named `tool_choice` requests for Heimdall actions.
- `--moe-backend triton` avoids FlashInfer TRTLLM MoE JIT for `Qwen3-30B-A3B`.

This is compatibility-oriented, not maximum-throughput Blackwell tuning. A
CUDA-toolkit-capable image or tuned vLLM container should be tested later to
remove `--enforce-eager` and restore FlashInfer/DeepGEMM paths.

## Tested Models

All tests used the OpenAI-compatible endpoint and named `propose_bid` tool
calls. The model-ladder checks below were historical smoke tests and used an 8k
served context. Current real AI-society experiment matrices use 16k serving
(`HEIMDALL_MAX_MODEL_LEN=16384`) with two independent vLLM endpoints:
`http://127.0.0.1:8000/v1` on GPU0 and `http://127.0.0.1:8001/v1` on GPU1.

Small ladder, five concurrent agents per loaded model:

| Model | Launch | Healthcheck | 5-agent named tool calls |
|---|---:|---:|---:|
| `Qwen/Qwen3-0.6B` | PASS | PASS | PASS |
| `Qwen/Qwen3-1.7B` | PASS | PASS | PASS |
| `Qwen/Qwen3-4B` | PASS | PASS | PASS |
| `Qwen/Qwen3-8B` | PASS | PASS | PASS |
| `Qwen/Qwen3-14B` | PASS | PASS | PASS |

Bigger ladder, one and five concurrent agents per loaded model:

| Model | Launch | Healthcheck | 1 agent | 5 agents | Observed VRAM |
|---|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-30B-A3B` | PASS | PASS | PASS | PASS | ~110.6 GiB |
| `Qwen/Qwen3-32B` | PASS | PASS | PASS | PASS | ~111.3 GiB |

Default repo config is `Qwen/Qwen3-32B` to prove the larger path. For lighter
daily development on UCloud, set:

```text
HEIMDALL_MODEL=Qwen/Qwen3-4B
HEIMDALL_GPU_MEMORY_UTILIZATION=0.45
```

## Operational Commands

```bash
cd /work/heimdall-vllm
bash scripts/launch_tmux.sh
tail -f logs/vllm.log
source .venv/bin/activate
python scripts/healthcheck_vllm.py
python tests/test_heimdall_named_tool_call.py
python tests/test_heimdall_n_agents.py --agents 1
python tests/test_heimdall_n_agents.py --agents 5
bash scripts/stop_vllm.sh
source scripts/export_env.sh
```

Optional model matrices:

```bash
bash scripts/run_model_matrix.sh
bash scripts/run_bigger_model_matrix.sh
```

## Heimdall Tool-Call Policy

- Prefer named `tool_choice` or `tool_choice="required"` for simulator,
  forecaster, and verifier calls.
- Avoid relying on `tool_choice="auto"` for critical actions unless the
  model-specific parser has been tested.
- Validate every returned action with Pydantic before passing it to the market
  simulator or verifier.
- Treat LLM output as a proposal, never as an accepted market action.
