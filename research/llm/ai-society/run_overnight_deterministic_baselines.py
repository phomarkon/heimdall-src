from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from heimdall_ai_society.config import load_config


CONTEXT_DIR = Path("data/cache/real_context/april_2026")
TRUTH_DIR = Path("data/cache/evaluation_truth/april_2026")
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
    parser = argparse.ArgumentParser(description="Run deterministic overnight baselines sequentially.")
    parser.add_argument("--config-list", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    configs = [Path(line.strip()) for line in args.config_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    _assert_configs(configs)
    results: list[dict[str, Any]] = []
    _write_json(args.log_dir / "results.json", results)

    for index, config in enumerate(configs, start=1):
        started = time.time()
        run_id = config.stem
        print(f"[{_now()}] {index}/{len(configs)} start {run_id} deterministic", flush=True)
        try:
            run_dir = _run_config(config, args.log_dir)
            _validate_trace(run_dir)
            eval_summary = _evaluate(run_dir)
            row = {
                "ok": True,
                "index": index,
                "config": str(config),
                "run_id": run_id,
                "run_dir": str(run_dir),
                "model": "deterministic_best_accepted",
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
                "model": "deterministic_best_accepted",
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


def _assert_configs(configs: list[Path]) -> None:
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
        if payload["chooser_mode"] != "deterministic_best_accepted":
            raise RuntimeError(f"bad chooser mode: {config}")
        if payload["ablation_strategy"] != "comm_broadcast_digest":
            raise RuntimeError(f"bad ablation strategy: {config}")
        if payload["llm"].get("enabled") is not False:
            raise RuntimeError(f"deterministic config must disable LLM: {config}")


def _run_config(config: Path, log_dir: Path) -> Path:
    run_id = config.stem
    out_path = log_dir / f"{run_id}.run.log"
    result = subprocess.run(
        ["uv", "run", "python", "-m", "heimdall_ai_society", "run", "--config", str(config)],
        env=_env(),
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
    if len(lines) != expected_rows:
        raise RuntimeError(f"trace row mismatch for {run_dir}: {len(lines)} != {expected_rows}")
    if summary.get("llm_enabled"):
        raise RuntimeError(f"deterministic run unexpectedly enabled LLM: {summary_path}")
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


def _now() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
