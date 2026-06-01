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

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONTEXT_DIR = Path("data/cache/real_context/april_2026")
TRUTH_DIR = Path("data/cache/evaluation_truth/april_2026")
EXPECTED_MODEL = "Qwen/Qwen3-32B"
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
    "asset_backend_disagreement_rate",
    "asset_backend_proxy_false_positive_rate",
    "asset_backend_scenario_envelope_false_positive_rate",
    "autonomous_tool_call_rate",
    "simulator_self_call_rate",
    "unsupported_bid_proposal_rate",
    "accepted_bid_backed_by_llm_requested_simulator_rate",
    "extra_tool_rounds_per_decision",
    "final_action_forced_rate",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one GPU split of the market-intelligence full suite.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--gpu", required=True, choices=["gpu0", "gpu1"])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--config-list", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    configs = [Path(line.strip()) for line in args.config_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    results: list[dict[str, Any]] = []
    _write_json(args.log_dir / f"{args.gpu}-results.json", results)
    _assert_served_model(args.base_url)
    _assert_unique_run_ids(configs, args.base_url)

    for index, config in enumerate(configs, start=1):
        started = time.time()
        run_id = config.stem
        print(f"[{_now()}] {args.gpu} {args.stage} {index}/{len(configs)} start {run_id}", flush=True)
        try:
            run_dir = _run_with_retry(config, args.base_url, args.log_dir)
            _validate_trace(run_dir)
            eval_summary = _evaluate(run_dir)
            row = {
                "ok": True,
                "stage": args.stage,
                "gpu": args.gpu,
                "config": str(config),
                "run_id": run_id,
                "run_dir": str(run_dir),
                "elapsed_seconds": round(time.time() - started, 3),
                "metrics": {key: eval_summary.get(key) for key in REPORT_KEYS},
            }
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        except Exception as exc:
            row = {
                "ok": False,
                "stage": args.stage,
                "gpu": args.gpu,
                "config": str(config),
                "run_id": run_id,
                "elapsed_seconds": round(time.time() - started, 3),
                "error": str(exc),
            }
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if not args.continue_on_failure:
                _write_json(args.log_dir / f"{args.gpu}-results.json", results)
                _write_summary(args.log_dir, args.gpu, args.stage, results)
                return 1
        _write_json(args.log_dir / f"{args.gpu}-results.json", results)
        _write_summary(args.log_dir, args.gpu, args.stage, results)
    return 0


def _run_with_retry(config: Path, base_url: str, log_dir: Path) -> Path:
    last_error: Exception | None = None
    for attempt in (1, 2):
        try:
            return _run_config(config, base_url, log_dir, attempt)
        except Exception as exc:
            last_error = exc
            print(f"[{_now()}] retryable failure attempt={attempt} config={config}: {exc}", flush=True)
            if attempt == 1:
                time.sleep(15)
    assert last_error is not None
    raise last_error


def _run_config(config: Path, base_url: str, log_dir: Path, attempt: int) -> Path:
    run_id = config.stem
    env = _env(base_url)
    out_path = log_dir / f"{run_id}.attempt{attempt}.run.log"
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
    if len(lines) != expected_rows:
        raise RuntimeError(f"trace row mismatch for {run_dir}: {len(lines)} != {expected_rows}")
    if summary.get("llm_enabled") and summary.get("chooser_mode", "llm") == "llm":
        if summary.get("llm_model_configured") != summary.get("llm_model_served"):
            raise RuntimeError(f"model mismatch in {summary_path}: {summary.get('llm_model_configured')} != {summary.get('llm_model_served')}")
    if summary.get("final_bid_guard") != "schema_only_shadow":
        breaches = _guard_breaches(lines)
        if breaches:
            raise RuntimeError(f"guard breaches in {trace_path}: {breaches[:3]}")
    else:
        shadow_missing = _shadow_missing(lines)
        if shadow_missing:
            raise RuntimeError(f"missing shadow simulations in {trace_path}: {shadow_missing[:3]}")
    if summary.get("safety_toolset") == "context_only":
        leaks = _safety_tool_leaks(lines)
        if leaks:
            raise RuntimeError(f"context-only safety tool leaks in {trace_path}: {leaks[:3]}")
    _validate_preprobe_mode(summary, lines, trace_path)
    for payload in lines:
        if payload["observed_at"] > payload["timestamp"]:
            raise RuntimeError(f"observed_at after timestamp in {trace_path}")
        if "LLM call failed" in str(payload.get("rationale", "")):
            raise RuntimeError(f"LLM failure fallback found in {trace_path}")


def _guard_breaches(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required_by_archetype = {
        "society_chair": "society_chair_consensus",
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
        required = required_by_archetype.get(str(row.get("archetype")))
        if required is None:
            breaches.append({"agent_id": row.get("agent_id"), "step": row.get("step"), "reason": "unsupported_bid_archetype"})
            continue
        found = False
        for call in row.get("tool_calls") or []:
            args = call.get("arguments") or {}
            result = call.get("result") or {}
            if required == "society_chair_consensus":
                selected = result.get("selected") or {}
                candidate = selected.get("candidate") or {}
                if (
                    call.get("name") == required
                    and call.get("ok") is True
                    and result.get("authority") == "deterministic_consensus"
                    and _same_float(candidate.get("quantity_mwh"), decision.get("quantity_mwh"))
                    and _same_float(candidate.get("limit_price_eur_mwh"), decision.get("limit_price_eur_mwh"))
                    and candidate.get("side") == decision.get("side")
                ):
                    found = True
                    break
                continue
            if (
                call.get("name") == required
                and call.get("ok") is True
                and (result.get("authority") == "authoritative" or result.get("controls_acceptance") is True)
                and result.get("accepted") is True
                and args.get("side") == decision.get("side")
                and _same_float(args.get("quantity_mwh"), decision.get("quantity_mwh"))
                and _same_float(args.get("limit_price_eur_mwh"), decision.get("limit_price_eur_mwh"))
            ):
                found = True
                break
        if not found:
            breaches.append({"agent_id": row.get("agent_id"), "step": row.get("step"), "archetype": row.get("archetype"), "decision": decision})
    return breaches


def _shadow_missing(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing = []
    for row in lines:
        decision = row.get("decision") or {}
        if decision.get("action") != "bid":
            continue
        if any((call.get("name") == "shadow_required_simulation") for call in row.get("tool_calls") or []):
            continue
        missing.append({"agent_id": row.get("agent_id"), "step": row.get("step"), "archetype": row.get("archetype"), "decision": decision})
    return missing


def _safety_tool_leaks(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leaks = []
    hidden_names = {
        "simulate_bid",
        "simulate_ev_bid",
        "simulate_wind_bid",
        "simulate_generator_bid",
        "simulate_retailer_bid",
        "simulate_renewables_bid",
        "candidate_menu",
        "rank_candidate_set",
        "get_limit_price_guidance",
        "get_candidate_rejection_summary",
        "get_candidate_sizing_guidance",
        "get_decision_trace_summary",
    }
    for row in lines:
        for call in row.get("tool_calls") or []:
            name = str(call.get("name") or "")
            if name == "shadow_required_simulation":
                continue
            if name in hidden_names or (name.startswith("get_") and name.endswith("_bid_feasibility")):
                leaks.append({"agent_id": row.get("agent_id"), "step": row.get("step"), "name": name})
    return leaks


def _validate_preprobe_mode(summary: dict[str, Any], lines: list[dict[str, Any]], trace_path: Path) -> None:
    mode = str(summary.get("preprobe_mode") or "full")
    if mode not in {"full", "context_only", "specialist_context", "none"}:
        raise RuntimeError(f"unknown preprobe_mode in {trace_path}: {mode}")
    seeded = _calls_with_provenance(lines, "runner_seeded")
    seeded_context = [call for call in seeded if str(call.get("name") or "") in _PREPROBE_CONTEXT_TOOLS]
    seeded_action = [call for call in seeded if _is_action_probe_call(str(call.get("name") or ""))]
    if mode == "full":
        if not seeded_context:
            raise RuntimeError(f"preprobe_mode=full produced no runner-seeded context calls in {trace_path}")
        if summary.get("safety_toolset") != "context_only" and not seeded_action:
            raise RuntimeError(f"preprobe_mode=full produced no runner-seeded action probes in {trace_path}")
    elif mode == "context_only":
        if not seeded_context:
            raise RuntimeError(f"preprobe_mode=context_only produced no runner-seeded context calls in {trace_path}")
        if seeded_action:
            raise RuntimeError(f"preprobe_mode=context_only leaked runner-seeded action probes in {trace_path}: {seeded_action[:3]}")
    elif mode == "none" and seeded:
        raise RuntimeError(f"preprobe_mode=none produced runner-seeded calls in {trace_path}: {seeded[:3]}")


_PREPROBE_CONTEXT_TOOLS = {
    "run_forecaster",
    "get_activation_context",
    "get_opportunity_context",
    "get_market_regime_context",
    "get_uncertainty_digest",
}


def _calls_with_provenance(lines: list[dict[str, Any]], provenance: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for row in lines:
        for call in row.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("provenance") == provenance:
                calls.append(call)
    return calls


def _is_action_probe_call(name: str) -> bool:
    return (
        name == "candidate_menu"
        or name == "rank_candidate_set"
        or name.startswith("simulate")
        or (name.startswith("get_") and name.endswith("_bid_feasibility"))
    )


def _same_float(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) < 1e-9
    except (TypeError, ValueError):
        return False


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
        ],
        env=_env("http://127.0.0.1:8000/v1"),
    )
    return payload["run_summary"]


def _assert_served_model(base_url: str) -> None:
    served = _served_model(base_url)
    if served != EXPECTED_MODEL:
        raise RuntimeError(f"served model mismatch at {base_url}: {served} != {EXPECTED_MODEL}")


def _assert_unique_run_ids(configs: list[Path], base_url: str) -> None:
    from heimdall_ai_society.config import load_config

    seen: set[str] = set()
    for config in configs:
        payload = load_config(config).model_dump(mode="json")
        run_id = str(payload["run_id"])
        if run_id in seen:
            raise RuntimeError(f"duplicate run_id in split: {run_id}")
        seen.add(run_id)
        if payload["market_context"] != "real" or payload["context_dataset_dir"] != str(CONTEXT_DIR):
            raise RuntimeError(f"bad context config: {config}")
        if payload.get("cache_refresh") is not False:
            raise RuntimeError(f"cache_refresh must be false: {config}")


def _served_model(base_url: str) -> str:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers={"Authorization": "Bearer heimdall-local"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data") or []
    if not data:
        raise RuntimeError(f"{base_url} /models returned no models")
    return str(data[0]["id"])


def _run_json(command: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(command, check=True, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = result.stdout.strip()
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"command did not emit JSON: {' '.join(command)}\n{text}")
    return json.loads(text[start:])


def _env(base_url: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "." if not env.get("PYTHONPATH") else f".:{env['PYTHONPATH']}"
    env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_API_KEY"] = "heimdall-local"
    return env


def _write_summary(log_dir: Path, gpu: str, stage: str, results: list[dict[str, Any]]) -> None:
    ok = [row for row in results if row.get("ok")]
    failed = [row for row in results if not row.get("ok")]
    payload = {
        "stage": stage,
        "gpu": gpu,
        "updated_at_utc": _now(),
        "completed": len(ok),
        "failed": len(failed),
        "latest_run_id": results[-1]["run_id"] if results else None,
        "failures": failed,
    }
    _write_json(log_dir / f"{gpu}-summary.json", payload)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
