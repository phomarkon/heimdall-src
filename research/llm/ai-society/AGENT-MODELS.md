# Heimdall Agent Model Notes

This document records the UCloud native vLLM model tests for Heimdall society
agents. Results are from a single NVIDIA B200 UCloud server using the
OpenAI-compatible vLLM endpoint, not vLLM Python APIs.

## Runtime Baseline

Validated runtime:

```text
OS: Ubuntu 24.04.4 LTS
GPU: NVIDIA B200, 183359 MiB VRAM, compute capability 10.0
Driver: 595.71.05
torch: 2.11.0+cu130
torch CUDA: 13.0
vLLM: 0.20.2
Endpoint: http://127.0.0.1:8000/v1
```

Known-good UCloud compatibility flags:

```text
HEIMDALL_ATTENTION_BACKEND=TRITON_ATTN
VLLM_USE_DEEP_GEMM=0
HEIMDALL_VLLM_EXTRA_ARGS="--enforce-eager --enable-auto-tool-choice --tool-call-parser qwen3_xml --moe-backend triton"
```

These flags avoid FlashInfer and DeepGEMM JIT paths that need `nvcc` or a
CUDA toolkit path not present in the tested UCloud native image.

## Model Comparison

The table combines sequential 32k-context reasoning/tool tests, the requested
Qwen2.5/DeepSeek/Mistral comparison, and the earlier 5/20/50/100-agent named
tool-call checks.

| Model | Loaded | Context | Reasoning trace | Reason latency | Auto tool | Required tool | Named tool | Verdict |
|---|---:|---:|---|---:|---|---|---|---|
| `Qwen/Qwen3-14B` | yes | 32k | visible raw `<think>` | 10.8s | PASS via fallback, `propose_bid` in fixed run | PASS `get_forecast` | PASS `propose_bid` | Recommended lighter dev/reasoning model |
| `Qwen/Qwen3-32B` | yes | 32k | visible raw `<think>` | 18.0s | PASS via fallback, `get_forecast` | PASS `get_forecast` | PASS `propose_bid` | Best Qwen first-real-simulator default |
| `QuixiAI/Qwen3-235B-A22B-AWQ` | yes | 32k | visible raw `<think>`, sometimes unclosed | 33.3s | PASS `verify_bid` | PASS `propose_bid` | PASS `propose_bid` | Strongest one-B200 candidate |
| `Qwen/Qwen2.5-72B-Instruct` | yes | 32k | none visible | 8.5s | PASS `verify_bid` via raw XML | PASS `get_forecast` | PASS `propose_bid` | Recommended action model, no visible thinking |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | yes | 32k | none visible in action run | 21.1s | NO_TOOL | NO_TOOL | PASS `propose_bid` | Reasoning-only candidate |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | yes | 32k | none visible in action run | 30.5s | NO_TOOL | NO_TOOL | PASS `propose_bid` | Reasoning-only candidate |
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | yes | 32k | none visible | 5.5s | PASS `get_forecast` | PASS `get_forecast` | PASS `propose_bid` | Recommended fast tool-action model |
| `QuixiAI/Qwen3-72B-Instruct-2` | yes | 32k | visible raw `<think>` | 8.4s | malformed raw tool text | NO_TOOL | weak PASS, poor arguments | Not suitable for Heimdall actions |
| `openai/gpt-oss-20b` | no | n/a | n/a | n/a | n/a | n/a | n/a | Needs CUDA-toolkit image/container |

## Operational Interpretation

- First real simulator run: use `Qwen/Qwen3-32B` when visible reasoning traces
  matter, or `mistralai/Mistral-Small-3.2-24B-Instruct-2506` when speed and
  clean OpenAI tool calls matter most.
- Strongest one-B200 model tested so far: `QuixiAI/Qwen3-235B-A22B-AWQ`. It
  gives stronger-looking reasoning and valid tool calls, but sequential
  reasoning calls were around 33 seconds at 32k context.
- Official dense 72B result: `Qwen/Qwen2.5-72B-Instruct` loaded on one B200 and
  passed auto, required, and named tool checks. It used about 135 GiB for model
  weights and only had about 3x full-context concurrency at 32k.
- DeepSeek R1 distills loaded and answered reasoning prompts, but failed
  autonomous `auto` and `required` tool-choice behavior in the action harness.
  Treat them as reasoning-only candidates unless paired with a separate
  action/tool model.
- `QuixiAI/Qwen3-72B-Instruct-2` fit in VRAM, but tool-call JSON quality was
  unreliable. Do not use it for market actions.
- `openai/gpt-oss-20b` failed on the native UCloud image because FP4/MXFP4
  loading tried to JIT FlashInfer kernels and required `nvcc` or
  `/usr/local/cuda`. Rerun it in a CUDA-toolkit-capable image or tuned vLLM
  container.

## Tool-Call Policy

- For market actions, prefer named `tool_choice` or `tool_choice="required"`.
- Do not rely on `tool_choice="auto"` for critical actions until the specific
  model/parser pair has passed the Heimdall harness.
- Keep reasoning and action generation as separate calls when needed:
  reasoning text first, structured tool call second.
- Validate every returned action with Pydantic and the verifier before passing
  it to the market simulator.
- Treat every LLM output as a proposal, never as an accepted market action.

## Source Logs

Raw logs are intentionally not committed. The tested UCloud server kept them
under `/work/heimdall-vllm/logs/`:

```text
/work/heimdall-vllm/logs/model_comparison_20260511-134125.summary.tsv
/work/heimdall-vllm/logs/reason_tool_context_20260511-130340.summary.tsv
/work/heimdall-vllm/logs/requested_agent_matrix_20260511-122033.summary.tsv
```
