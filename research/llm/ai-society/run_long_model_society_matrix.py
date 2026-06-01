from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from heimdall_ai_society.config import load_config

CONTEXT_DIR = Path("data/cache/real_context/april_2026")
TRUTH_DIR = Path("data/cache/evaluation_truth/april_2026")
VLLM_DIR = Path("/work/heimdall-vllm")
ENDPOINTS = [
    {"gpu": "0", "port": "8000", "session": "heimdall-vllm-long-gpu0"},
    {"gpu": "1", "port": "8001", "session": "heimdall-vllm-long-gpu1"},
]
OLD_VLLM_SESSIONS = ["heimdall-vllm", "heimdall-vllm-gpu1"]
REPORT_KEYS = [
    "must_watch_profitable_or_activated_recall",
    "must_watch_profitable_or_activated_precision",
    "profitable_watch_or_bid_recall",
    "alert_spam_rate",
    "contested_watch_count",
    "reason_diversity",
    "evidence_reference_validity",
    "confidence_dispersion",
    "realized_profit_eur",
    "truth_window_oracle_capture",
    "verifier_realized_profit_breach_rate",
    "wrong_side_count",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the long dual-GPU society matrix sequentially.")
    parser.add_argument("--config-list", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--skip-vllm-restart", action="store_true")
    parser.add_argument("--health-timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    configs = [Path(line.strip()) for line in args.config_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    _assert_unique_run_ids(configs)
    results: list[dict[str, Any]] = []
    _write_json(args.log_dir / "results.json", results)

    active_model: str | None = None
    for index, config in enumerate(configs, start=1):
        started = time.time()
        run_id = config.stem
        payload = load_config(config).model_dump(mode="json")
        expected_model = str(payload["llm"]["model"])
        print(f"[{_now()}] {index}/{len(configs)} start {run_id} model={expected_model}", flush=True)
        try:
            if active_model != expected_model:
                if not args.skip_vllm_restart:
                    _restart_vllm(expected_model, args.log_dir, args.health_timeout_seconds)
                _assert_served_models(expected_model)
                active_model = expected_model
            else:
                _assert_served_models(expected_model)
            run_dir = _run_config(config, args.log_dir)
            _validate_trace(run_dir)
            eval_summary = _evaluate(run_dir)
            row = {
                "ok": True,
                "index": index,
                "config": str(config),
                "run_id": run_id,
                "run_dir": str(run_dir),
                "model": expected_model,
                "elapsed_seconds": round(time.time() - started, 3),
                "metrics": {key: eval_summary.get(key) for key in REPORT_KEYS},
            }
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        except Exception as exc:
            row = {
                "ok": False,
                "index": index,
                "config": str(config),
                "run_id": run_id,
                "model": expected_model,
                "elapsed_seconds": round(time.time() - started, 3),
                "error": str(exc),
            }
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if not args.continue_on_failure:
                _write_json(args.log_dir / "results.json", results)
                _write_summary(args.log_dir, results)
                return 1
        _write_json(args.log_dir / "results.json", results)
        _write_summary(args.log_dir, results)
    return 0


def _restart_vllm(model: str, log_dir: Path, health_timeout_seconds: int) -> None:
    for session in [*OLD_VLLM_SESSIONS, *(endpoint["session"] for endpoint in ENDPOINTS)]:
        subprocess.run(["tmux", "kill-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Also free the ports in case vLLM was started outside our tmux sessions
    # (e.g. by an earlier interactive session). Without this the new vLLM hits
    # OOM trying to share a GPU with the legacy process and silently exits.
    for endpoint in ENDPOINTS:
        subprocess.run(
            ["bash", "-c", f"lsof -ti tcp:{endpoint['port']} | xargs -r kill -9"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    time.sleep(3)  # let the GPUs release VRAM before the new vLLM allocates
    for endpoint in ENDPOINTS:
        port = endpoint["port"]
        gpu = endpoint["gpu"]
        session = endpoint["session"]
        base_dir = _prepare_vllm_base(model, port, gpu)
        log_path = (log_dir / f"vllm-gpu{gpu}-{_model_slug(model)}.log").resolve()
        command = (
            f"cd {VLLM_DIR} && "
            f"CUDA_VISIBLE_DEVICES={gpu} "
            f"HEIMDALL_VLLM_BASE_DIR={_shell_quote(str(base_dir))} "
            f"bash scripts/start_vllm.sh > {_shell_quote(str(log_path))} 2>&1"
        )
        subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)
    deadline = time.time() + health_timeout_seconds
    while time.time() < deadline:
        try:
            _assert_served_models(model)
            return
        except Exception as exc:
            print(f"[{_now()}] waiting for vLLM model={model}: {exc}", flush=True)
            time.sleep(15)
    _assert_served_models(model)


def _prepare_vllm_base(model: str, port: str, gpu: str) -> Path:
    base_dir = Path(f"/tmp/heimdall-vllm-long-gpu{gpu}")
    base_dir.mkdir(parents=True, exist_ok=True)
    venv_link = base_dir / ".venv"
    if not venv_link.exists():
        venv_link.symlink_to(VLLM_DIR / ".venv", target_is_directory=True)
    (base_dir / "logs").mkdir(exist_ok=True)
    env_text = "\n".join(
        [
            f"HEIMDALL_MODEL={model}",
            "HEIMDALL_VLLM_HOST=127.0.0.1",
            f"HEIMDALL_VLLM_PORT={port}",
            "HEIMDALL_VLLM_API_KEY=heimdall-local",
            "HEIMDALL_TENSOR_PARALLEL_SIZE=1",
            "HEIMDALL_MAX_MODEL_LEN=16384",
            f"HEIMDALL_GPU_MEMORY_UTILIZATION={_gpu_memory_utilization(model)}",
            "HEIMDALL_DTYPE=auto",
            "HEIMDALL_TRUST_REMOTE_CODE=0",
            "HEIMDALL_ATTENTION_BACKEND=TRITON_ATTN",
            "VLLM_USE_DEEP_GEMM=0",
            # TRITON_ATTN / flashinfer JIT-compile per shape and need nvcc when
            # the kernel isn't cached. B200 is Blackwell (sm_100a), needs CUDA 12.8+.
            # NVHPC 25.3 ships CUDA 12.8 but splits math libraries (curand etc.)
            # into a parallel math_libs/12.8 tree, so CPATH + LD_LIBRARY_PATH must
            # be extended for flashinfer's nvcc invocation to find curand.h.
            "CUDA_HOME=/opt/easybuild/ubuntu-24.04/amd/software/NVHPC/25.3-CUDA-12.8.0/Linux_x86_64/25.3/cuda/12.8",
            "PATH=/opt/easybuild/ubuntu-24.04/amd/software/NVHPC/25.3-CUDA-12.8.0/Linux_x86_64/25.3/cuda/12.8/bin:$PATH",
            "CPATH=/opt/easybuild/ubuntu-24.04/amd/software/NVHPC/25.3-CUDA-12.8.0/Linux_x86_64/25.3/math_libs/12.8/targets/x86_64-linux/include:${CPATH:-}",
            "LD_LIBRARY_PATH=/opt/easybuild/ubuntu-24.04/amd/software/NVHPC/25.3-CUDA-12.8.0/Linux_x86_64/25.3/math_libs/12.8/targets/x86_64-linux/lib:/opt/easybuild/ubuntu-24.04/amd/software/NVHPC/25.3-CUDA-12.8.0/Linux_x86_64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}",
            'HEIMDALL_VLLM_EXTRA_ARGS="--enforce-eager --enable-auto-tool-choice --tool-call-parser qwen3_xml --moe-backend triton"',
            "",
        ]
    )
    (base_dir / ".env").write_text(env_text, encoding="utf-8")
    return base_dir


def _gpu_memory_utilization(model: str) -> str:
    if any(size in model for size in ["72B", "80B", "120B"]):
        return "0.90"
    return "0.60"


def _run_config(config: Path, log_dir: Path) -> Path:
    run_id = config.stem
    env = _env()
    out_path = log_dir / f"{run_id}.run.log"
    result = subprocess.run(
        ["uv", "run", "python", "-m", "heimdall_ai_society", "run", "--config", str(config)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"run failed rc={result.returncode}; log={out_path}")
    for line in result.stdout.splitlines():
        if line.startswith("wrote society run:"):
            return Path(line.split(":", 1)[1].strip())
    raise RuntimeError(f"could not parse run dir from {out_path}")


def _validate_trace(run_dir: Path) -> None:
    trace_path = run_dir / "traces.jsonl"
    summary_path = run_dir / "summary.json"
    if not trace_path.exists() or not summary_path.exists():
        raise RuntimeError(f"missing run artifacts in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_rows = int(summary["ticks"]) * int(summary["agent_count"])
    if str(summary.get("ablation_strategy", "")).startswith("comm_society_chair"):
        expected_rows += int(summary["ticks"])
    if summary.get("ablation_strategy") == "comm_central_supervisor":
        expected_rows += int(summary["ticks"])
    if len(lines) != expected_rows:
        raise RuntimeError(f"trace row mismatch for {run_dir}: {len(lines)} != {expected_rows}")
    if summary.get("llm_enabled") and summary.get("chooser_mode", "llm") == "llm":
        served = summary.get("llm_model_served")
        configured = summary.get("llm_model_configured")
        served_many = set((summary.get("llm_models_served") or {}).values())
        if served != configured and served_many != {configured}:
            raise RuntimeError(f"model mismatch in {summary_path}: {served=} {served_many=} {configured=}")
    for payload in lines:
        if payload["observed_at"] > payload["timestamp"]:
            raise RuntimeError(f"observed_at after timestamp in {trace_path}")
        if "LLM call failed" in str(payload.get("rationale", "")):
            raise RuntimeError(f"LLM failure fallback found in {trace_path}")


def _evaluate(run_dir: Path) -> dict[str, Any]:
    output_dir = Path("evaluations") / run_dir.name
    payload = _run_json(
        [
            "uv",
            "run",
            "python",
            "tools/evaluation/evaluate_society_run.py",
            "--run-dir",
            str(run_dir),
            "--context-dir",
            str(CONTEXT_DIR),
            "--truth-dir",
            str(TRUTH_DIR),
            "--output-dir",
            str(output_dir),
        ]
    )
    return payload["run_summary"]


def _assert_unique_run_ids(configs: list[Path]) -> None:
    seen: set[str] = set()
    for config in configs:
        payload = load_config(config).model_dump(mode="json")
        run_id = str(payload["run_id"])
        if run_id in seen:
            raise RuntimeError(f"duplicate run_id: {run_id}")
        seen.add(run_id)
        if payload["market_context"] != "real" or payload["context_dataset_dir"] != str(CONTEXT_DIR):
            raise RuntimeError(f"bad context config: {config}")
        if payload.get("cache_refresh") is not False:
            raise RuntimeError(f"cache_refresh must be false: {config}")
        if payload["llm"].get("base_urls") != ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"]:
            raise RuntimeError(f"config must use both local vLLM endpoints: {config}")


def _assert_served_models(expected_model: str) -> None:
    served = {endpoint["port"]: _served_model(f"http://127.0.0.1:{endpoint['port']}/v1") for endpoint in ENDPOINTS}
    mismatched = {port: model for port, model in served.items() if model != expected_model}
    if mismatched:
        raise RuntimeError(f"served model mismatch: expected={expected_model} served={served}")


def _served_model(base_url: str) -> str:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers={"Authorization": "Bearer heimdall-local"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data") or []
    if not data:
        raise RuntimeError(f"{base_url} /models returned no models")
    return str(data[0]["id"])


def _run_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, check=True, env=_env(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = result.stdout.strip()
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"command did not emit JSON: {' '.join(command)}\n{text}")
    return json.loads(text[start:])


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "." if not env.get("PYTHONPATH") else f".:{env['PYTHONPATH']}"
    env["OPENAI_API_KEY"] = "heimdall-local"
    return env


def _write_summary(log_dir: Path, results: list[dict[str, Any]]) -> None:
    ok = [row for row in results if row.get("ok")]
    failed = [row for row in results if not row.get("ok")]
    payload = {
        "updated_at_utc": _now(),
        "completed": len(ok),
        "failed": len(failed),
        "latest_run_id": results[-1]["run_id"] if results else None,
        "failures": failed,
    }
    _write_json(log_dir / "summary.json", payload)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_slug(model: str) -> str:
    return model.split("/")[-1].lower().replace(".", "").replace("-", "")


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _now() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
