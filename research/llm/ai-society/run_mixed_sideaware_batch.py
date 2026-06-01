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

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONFIG_DIR = Path("ai-society/configs/mixed-sideaware-20260515")
LOG_DIR = Path("ai-society/run_logs/mixed-sideaware-20260515")
CONTEXT_DIR = Path("data/cache/real_context/april_2026")
TRUTH_DIR = Path("data/cache/evaluation_truth/april_2026")
MEMORY_BANK = Path("ai-society/runs/risk-filter-matrix/memory-v2-bank.jsonl")
EXPECTED_MODEL = "Qwen/Qwen3-32B"

SMOKE_WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
}
SCREEN_WINDOWS = {
    "apr02-0530": "2026-04-02T05:30:00Z",
    "apr09-1830": "2026-04-09T18:30:00Z",
    "apr13-0015": "2026-04-13T00:15:00Z",
    "apr03-1430": "2026-04-03T14:30:00Z",
    "apr05-1030": "2026-04-05T10:30:00Z",
    "apr22-0830": "2026-04-22T08:30:00Z",
}
SCREEN_PROFILES = {
    "mixed18": ("mixed_expert_18_sideaware", 18),
    "mixed20": ("mixed_expert_20_sideaware", 20),
    "jao12": ("jao_grid_v1", 12),
    "info14": ("action_core_8_plus_info_specialists", 14),
    "core10": ("action_core_10_safety", 10),
    "balanced12": ("balanced_intelligence", 12),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and run mixed side-aware society batch.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--wait-command", default="bcast-mem-ext-promotion48")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--stage", choices=["all", "smoke", "screen"], default="all")
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifests = _write_configs()
    if args.prepare_only:
        print(json.dumps({"prepared": manifests}, indent=2, sort_keys=True))
        return 0

    _wait_for_old_matrix(args.wait_command, args.poll_seconds)
    _assert_jao_fixture()
    _assert_served_models()

    if args.stage in {"all", "smoke"}:
        smoke_results = _run_list(manifests["smoke"], "smoke", continue_on_failure=args.continue_on_failure)
        if not all(row.get("ok") for row in smoke_results):
            print(f"[{_now()}] smoke failed; not starting screen", flush=True)
            return 1
        _assert_mixed_smoke_success(smoke_results)
    if args.stage in {"all", "screen"}:
        _run_list(manifests["screen"], "screen", continue_on_failure=args.continue_on_failure)
    return 0


def _write_configs() -> dict[str, str]:
    smoke_paths = []
    for short, start in SMOKE_WINDOWS.items():
        for label, (profile, count) in {
            "mixed18": ("mixed_expert_18_sideaware", 18),
            "mixed20": ("mixed_expert_20_sideaware", 20),
        }.items():
            path = CONFIG_DIR / f"msa-smoke-{label}-{short}-2.yaml"
            _write_yaml(path, _config_payload(path.stem, profile, count, start, ticks=2))
            smoke_paths.append(path)

    screen_paths = []
    for short, start in SCREEN_WINDOWS.items():
        for label, (profile, count) in SCREEN_PROFILES.items():
            path = CONFIG_DIR / f"msa-screen-{label}-{short}-24.yaml"
            _write_yaml(path, _config_payload(path.stem, profile, count, start, ticks=24))
            screen_paths.append(path)

    smoke_manifest = CONFIG_DIR / "smoke.txt"
    screen_manifest = CONFIG_DIR / "screen.txt"
    smoke_manifest.write_text("\n".join(str(path) for path in smoke_paths) + "\n", encoding="utf-8")
    screen_manifest.write_text("\n".join(str(path) for path in screen_paths) + "\n", encoding="utf-8")
    return {"smoke": str(smoke_manifest), "screen": str(screen_manifest)}


def _config_payload(run_id: str, profile: str, agent_count: int, start: str, *, ticks: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": 42,
        "zone": "DK1",
        "agent_count": agent_count,
        "ticks": ticks,
        "start_timestamp": start,
        "forecaster_backend": "f8",
        "chooser_mode": "llm",
        "verifier_mode": "simulator",
        "verifier_tau_eur": 0.0,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "objective": "bid_seeking",
        "ablation_strategy": "comm_broadcast_digest",
        "persona_profile": profile,
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "p2h_only_simulator",
        "max_tool_rounds": 6,
        "simulator_max_concurrency": 8,
        "data_start": "2026-04-01T00:00:00Z",
        "data_end": "2026-05-01T00:00:00Z",
        "context_dataset_dir": str(CONTEXT_DIR),
        "data_cache_dir": str(CONTEXT_DIR / "source_cache"),
        "default_lookback_hours": 24,
        "cache_refresh": False,
        "output_dir": "ai-society/runs/mixed-sideaware-20260515",
        "memory_enabled": MEMORY_BANK.exists(),
        "memory_bank_path": str(MEMORY_BANK),
        "memory_max_items_per_agent": 5,
        "memory_max_prompt_chars": 2400,
        "reviewer_mode": "code_only",
        "llm": {
            "enabled": True,
            "model": EXPECTED_MODEL,
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "temperature": 0.2,
            "max_tokens": 512,
            "timeout_seconds": 180,
            "max_concurrency": 12,
        },
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _wait_for_old_matrix(pattern: str, poll_seconds: int) -> None:
    while True:
        result = subprocess.run(
            ["pgrep", "-af", pattern],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        lines = [line for line in result.stdout.splitlines() if "run_mixed_sideaware_batch.py" not in line]
        if not lines:
            print(f"[{_now()}] old matrix no longer running; starting mixed side-aware batch", flush=True)
            return
        print(f"[{_now()}] waiting for old matrix; active={len(lines)} next_check={poll_seconds}s", flush=True)
        time.sleep(poll_seconds)


def _run_list(manifest: str, stage: str, *, continue_on_failure: bool) -> list[dict[str, Any]]:
    configs = [Path(line.strip()) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    results: list[dict[str, Any]] = []
    _write_json(LOG_DIR / f"{stage}-results.json", results)
    for index, config in enumerate(configs, start=1):
        started = time.time()
        print(f"[{_now()}] {stage} {index}/{len(configs)} start {config.stem}", flush=True)
        try:
            _validate_config(config)
            run_dir = _run_config(config, stage)
            _validate_run(run_dir)
            eval_summary = _evaluate(run_dir)
            row = {
                "ok": True,
                "stage": stage,
                "config": str(config),
                "run_id": config.stem,
                "run_dir": str(run_dir),
                "elapsed_seconds": round(time.time() - started, 3),
                "metrics": _select_metrics(eval_summary),
            }
        except Exception as exc:
            row = {
                "ok": False,
                "stage": stage,
                "config": str(config),
                "run_id": config.stem,
                "elapsed_seconds": round(time.time() - started, 3),
                "error": str(exc),
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            results.append(row)
            _write_json(LOG_DIR / f"{stage}-results.json", results)
            if not continue_on_failure:
                return results
            continue
        print(json.dumps(row, sort_keys=True), flush=True)
        results.append(row)
        _write_json(LOG_DIR / f"{stage}-results.json", results)
    return results


def _validate_config(config: Path) -> None:
    _run(
        ["uv", "run", "python", "-m", "heimdall_ai_society", "validate-config", str(config)],
        log_path=LOG_DIR / f"{config.stem}.validate.log",
    )


def _run_config(config: Path, stage: str) -> Path:
    result = _run(
        ["uv", "run", "python", "-m", "heimdall_ai_society", "run", "--config", str(config)],
        log_path=LOG_DIR / f"{config.stem}.run.log",
    )
    for line in result.stdout.splitlines():
        if line.startswith("wrote society run:"):
            return Path(line.split(":", 1)[1].strip())
    raise RuntimeError(f"could not parse run dir for {config} in stage {stage}")


def _validate_run(run_dir: Path) -> None:
    trace_path = run_dir / "traces.jsonl"
    summary_path = run_dir / "summary.json"
    if not trace_path.exists() or not summary_path.exists():
        raise RuntimeError(f"missing artifacts in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = int(summary["ticks"]) * int(summary["agent_count"])
    if len(lines) != expected:
        raise RuntimeError(f"trace row mismatch {len(lines)} != {expected} for {run_dir}")
    if summary.get("forecast_routing_warnings") is None:
        raise RuntimeError(f"missing forecast routing warnings key in {summary_path}")
    _assert_no_guard_breaches(lines)
    if summary.get("persona_profile", "").startswith("mixed_expert"):
        backends = {row.get("forecast_backend") for row in lines if row.get("agent_id") != "society-chair"}
        if not {"f8", "f7", "f3_ensemble"}.issubset(backends):
            raise RuntimeError(f"mixed run missing backend diversity: {backends}")
    jao_rows = [
        call.get("result", {})
        for row in lines
        for call in row.get("tool_calls") or []
        if call.get("name") == "get_grid_constraints"
    ]
    if jao_rows and not any(result.get("authority") == "jao_derived_non_leaking" and int(result.get("row_count") or 0) > 0 for result in jao_rows):
        raise RuntimeError(f"JAO calls found but no active non-leaking JAO rows in {run_dir}")
    if any("LLM call failed" in str(row.get("rationale", "")) for row in lines):
        raise RuntimeError(f"LLM failure fallback found in {trace_path}")


def _assert_no_guard_breaches(lines: list[dict[str, Any]]) -> None:
    required = {
        "p2h": "simulate_bid",
        "ev": "simulate_ev_bid",
        "wind": "simulate_wind_bid",
        "generator": "simulate_generator_bid",
        "renewables": "simulate_renewables_bid",
        "retailer": "simulate_retailer_bid",
    }
    breaches = []
    for row in lines:
        decision = row.get("decision") or {}
        if decision.get("action") != "bid":
            continue
        tool_name = required.get(str(row.get("archetype")))
        if tool_name is None:
            breaches.append({"agent_id": row.get("agent_id"), "reason": "unsupported_bid_archetype"})
            continue
        found = False
        for call in row.get("tool_calls") or []:
            args = call.get("arguments") or {}
            result = call.get("result") or {}
            if (
                call.get("name") == tool_name
                and (result.get("authority") == "authoritative" or result.get("controls_acceptance") is True)
                and result.get("accepted") is True
                and args.get("side") == decision.get("side")
                and _same_float(args.get("quantity_mwh"), decision.get("quantity_mwh"))
                and _same_float(args.get("limit_price_eur_mwh"), decision.get("limit_price_eur_mwh"))
            ):
                found = True
                break
        if not found:
            breaches.append({"agent_id": row.get("agent_id"), "step": row.get("step"), "decision": decision})
    if breaches:
        raise RuntimeError(f"guard breaches: {breaches[:3]}")


def _evaluate(run_dir: Path) -> dict[str, Any]:
    output_dir = Path("evaluations") / run_dir.name
    result = _run(
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
        ],
        log_path=LOG_DIR / f"{run_dir.name}.eval.log",
    )
    text = result.stdout.strip()
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"evaluation did not emit JSON for {run_dir}")
    return json.loads(text[start:])["run_summary"]


def _select_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "realized_profit_eur",
        "wrong_side_count",
        "bid_action_count",
        "must_watch_profitable_or_activated_recall",
        "must_watch_profitable_or_activated_precision",
        "profitable_bid_precision",
        "profitable_bid_recall",
        "verifier_realized_profit_breach_rate",
    ]
    return {key: summary.get(key) for key in keys}


def _assert_mixed_smoke_success(results: list[dict[str, Any]]) -> None:
    for row in results:
        run_dir = Path(str(row["run_dir"]))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if str(summary.get("persona_profile", "")).startswith("mixed_expert"):
            backends = set(summary.get("forecast_backend_by_agent", {}).values())
            if not {"f8", "f7", "f3_ensemble"}.issubset(backends):
                raise RuntimeError(f"smoke mixed run missing backend routing: {run_dir}")


def _assert_jao_fixture() -> None:
    path = CONTEXT_DIR / "jao_constraints.parquet"
    if not path.exists() or path.stat().st_size < 1_000_000:
        raise RuntimeError(f"JAO fixture missing or too small: {path}")


def _assert_served_models() -> None:
    for base_url in ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"]:
        payload = json.loads(
            _run(
                ["curl", "-fsS", "-H", "Authorization: Bearer heimdall-local", f"{base_url}/models"],
                log_path=LOG_DIR / f"models-{base_url.split(':')[-1].replace('/', '')}.log",
            ).stdout
        )
        served = str((payload.get("data") or [{}])[0].get("id"))
        if served != EXPECTED_MODEL:
            raise RuntimeError(f"served model mismatch at {base_url}: {served} != {EXPECTED_MODEL}")


def _run(command: list[str], *, log_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "." if not env.get("PYTHONPATH") else f".:{env['PYTHONPATH']}"
    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"command failed rc={result.returncode}: {' '.join(command)}; log={log_path}")
    return result


def _same_float(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) < 1e-9
    except (TypeError, ValueError):
        return False


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
