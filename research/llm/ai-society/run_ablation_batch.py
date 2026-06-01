from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


REAL_CONTEXT_ROOT = Path("data/cache/real_context")
TRUTH_ROOT = Path("data/cache/evaluation_truth")
ALLOWED_CONTEXTS = {"april_2026", "2026_03"}


def _dirs_for(context_dataset_dir: str) -> tuple[Path, Path]:
    """Map a config's context_dataset_dir to its (context_dir, evaluation_truth_dir)."""
    name = Path(context_dataset_dir).name
    if name not in ALLOWED_CONTEXTS:
        raise RuntimeError(f"unknown context dataset {context_dataset_dir!r}; allowed: {sorted(ALLOWED_CONTEXTS)}")
    return REAL_CONTEXT_ROOT / name, TRUTH_ROOT / name


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Heimdall society ablation configs sequentially with guardrails.")
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()

    results = []
    for config in args.configs:
        started = time.time()
        try:
            run_id = config.stem
            validate_config(config, args.expected_model)
            payload = run_json(["uv", "run", "python", "-m", "heimdall_ai_society", "validate-config", str(config)])
            if payload.get("llm", {}).get("enabled", True) and payload.get("chooser_mode", "llm") == "llm":
                served = served_model(args.base_url)
                if served != args.expected_model:
                    raise RuntimeError(f"served model mismatch: expected {args.expected_model!r}, got {served!r}")
            run_dir = run_config(config)
            validate_trace(run_dir)
            context_dir, truth_dir = _dirs_for(payload["context_dataset_dir"])
            eval_summary = evaluate(run_dir, context_dir, truth_dir)
            elapsed = round(time.time() - started, 3)
            results.append({"config": str(config), "run_dir": str(run_dir), "elapsed_seconds": elapsed, "evaluation": eval_summary})
            print(json.dumps(results[-1], indent=2, sort_keys=True), flush=True)
        except Exception as exc:
            failure = {"config": str(config), "error": str(exc)}
            results.append(failure)
            print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
            if not args.continue_on_failure:
                write_batch_summary(results)
                return 1
    write_batch_summary(results)
    return 0


def served_model(base_url: str) -> str:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers={"Authorization": "Bearer heimdall-local"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data") or []
    if not data:
        raise RuntimeError("vLLM /models returned no models")
    return str(data[0]["id"])


def validate_config(config: Path, expected_model: str) -> None:
    payload = run_json(["uv", "run", "python", "-m", "heimdall_ai_society", "validate-config", str(config)])
    if payload.get("llm", {}).get("enabled", True) and payload.get("chooser_mode", "llm") == "llm" and payload["llm"]["model"] != expected_model:
        raise RuntimeError(f"{config} config model mismatch: {payload['llm']['model']} != {expected_model}")
    if payload["market_context"] != "real":
        raise RuntimeError(f"{config} must use real market_context")
    if Path(payload["context_dataset_dir"]).name not in ALLOWED_CONTEXTS:
        raise RuntimeError(f"{config} must use a known real_context dataset ({sorted(ALLOWED_CONTEXTS)})")
    if "evaluation_truth" in json.dumps(payload, sort_keys=True):
        raise RuntimeError(f"{config} leaks evaluation truth into run config")


def run_config(config: Path) -> Path:
    result = subprocess.run(
        ["uv", "run", "python", "-m", "heimdall_ai_society", "run", "--config", str(config)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, flush=True)
    for line in result.stdout.splitlines():
        if line.startswith("wrote society run:"):
            return Path(line.split(":", 1)[1].strip())
    raise RuntimeError(f"could not parse run dir from output for {config}")


def validate_trace(run_dir: Path) -> None:
    trace_path = run_dir / "traces.jsonl"
    summary_path = run_dir / "summary.json"
    if not trace_path.exists() or not summary_path.exists():
        raise RuntimeError(f"missing run artifacts in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_rows = int(summary["ticks"]) * int(summary["agent_count"])
    actual_rows = sum(1 for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if actual_rows != expected_rows:
        raise RuntimeError(f"trace row mismatch for {run_dir}: {actual_rows} != {expected_rows}")
    if summary.get("llm_enabled") and summary.get("chooser_mode", "llm") == "llm" and summary.get("llm_model_configured") != summary.get("llm_model_served"):
        raise RuntimeError(f"run model mismatch in {summary_path}: {summary.get('llm_model_configured')} != {summary.get('llm_model_served')}")
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload["observed_at"] > payload["timestamp"]:
            raise RuntimeError(f"observed_at after timestamp in {trace_path}")
        if "LLM call failed" in str(payload.get("rationale", "")):
            raise RuntimeError(f"LLM failure fallback found in {trace_path}")


def evaluate(run_dir: Path, context_dir: Path, truth_dir: Path) -> dict:
    output_dir = Path("evaluations") / run_dir.name
    payload = run_json(
        [
            "uv",
            "run",
            "python",
            "tools/evaluation/evaluate_society_run.py",
            "--run-dir",
            str(run_dir),
            "--context-dir",
            str(context_dir),
            "--truth-dir",
            str(truth_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    return payload["run_summary"]


def run_json(command: list[str]) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = "." if not env.get("PYTHONPATH") else f".:{env['PYTHONPATH']}"
    result = subprocess.run(command, check=True, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = result.stdout.strip()
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"command did not emit JSON: {' '.join(command)}\n{text}")
    return json.loads(text[start:])


def write_batch_summary(results: list[dict]) -> None:
    out = Path("ai-society/runs/ablation-batch-summary.json")
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
