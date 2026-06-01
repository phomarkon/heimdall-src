from __future__ import annotations

import json
from pathlib import Path
from typing import Any


WINDOWS = ("apr02-0530", "apr09-1830", "apr13-0015")
OUTPUT_DIR = Path("evaluations/verifierless-baseline-20260519")
RUN_ROOT = Path("ai-society/runs/verifierless-baseline-20260519")

METRICS = [
    "realized_profit_eur",
    "truth_window_oracle_capture",
    "bid_action_count",
    "watch_count",
    "wrong_side_count",
    "max_drawdown_eur",
    "downside_cvar_95_eur",
    "unsupported_bid_proposal_rate",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for window in WINDOWS:
        rows.extend(_s06_rows(window))
        rows.extend(_mixed20_rows(window))
    payload = {
        "matrix": "verifierless-baseline-20260519",
        "rows": rows,
        "shadow_rejection_examples": _shadow_rejection_examples(rows)[:40],
    }
    (OUTPUT_DIR / "paired_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _s06_rows(window: str) -> list[dict[str, Any]]:
    guarded = f"tsa-s06-scenario-full-{window}-seed42-q32"
    return [
        _comparison_row(
            society="s06-actioncore",
            window=window,
            guarded_run_id=guarded,
            variant="shadow-toolvisible",
            run_id=f"vlb-s06-shadow-toolvisible-{window}-seed42-q32",
        ),
        _comparison_row(
            society="s06-actioncore",
            window=window,
            guarded_run_id=guarded,
            variant="shadow-contextonly",
            run_id=f"vlb-s06-shadow-contextonly-{window}-seed42-q32",
        ),
    ]


def _mixed20_rows(window: str) -> list[dict[str, Any]]:
    guarded = f"vlb-mixed20-guarded-{window}-seed42-q32"
    return [
        _standalone_row("mixed20", window, "guarded", guarded),
        _comparison_row(
            society="mixed20",
            window=window,
            guarded_run_id=guarded,
            variant="shadow-toolvisible",
            run_id=f"vlb-mixed20-shadow-toolvisible-{window}-seed42-q32",
        ),
        _comparison_row(
            society="mixed20",
            window=window,
            guarded_run_id=guarded,
            variant="shadow-contextonly",
            run_id=f"vlb-mixed20-shadow-contextonly-{window}-seed42-q32",
        ),
    ]


def _standalone_row(society: str, window: str, variant: str, run_id: str) -> dict[str, Any]:
    summary = _summary(run_id)
    return {
        "society": society,
        "window": window,
        "variant": variant,
        "run_id": run_id,
        "metrics": {key: summary.get(key) for key in METRICS},
        "action_mix": _action_mix(summary),
        "shadow": _shadow_metrics(run_id),
        "context_only_tool_leak_count": _context_only_tool_leak_count(run_id) if variant == "shadow-contextonly" else None,
    }


def _comparison_row(society: str, window: str, guarded_run_id: str, variant: str, run_id: str) -> dict[str, Any]:
    guarded = _summary(guarded_run_id)
    unguarded = _summary(run_id)
    return {
        "society": society,
        "window": window,
        "variant": variant,
        "run_id": run_id,
        "guarded_run_id": guarded_run_id,
        "metrics": {
            key: {
                "guarded": guarded.get(key),
                "variant": unguarded.get(key),
                "delta": _delta(unguarded.get(key), guarded.get(key)),
            }
            for key in METRICS
        },
        "action_mix": {
            "guarded": _action_mix(guarded),
            "variant": _action_mix(unguarded),
        },
        "shadow": _shadow_metrics(run_id),
        "context_only_tool_leak_count": _context_only_tool_leak_count(run_id) if variant == "shadow-contextonly" else None,
    }


def _summary(run_id: str) -> dict[str, Any]:
    path = Path("evaluations") / run_id / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _traces(run_id: str) -> list[dict[str, Any]]:
    paths = [
        RUN_ROOT / run_id / "traces.jsonl",
        Path("ai-society/runs/thesis-s06-equal-ablation-20260519") / run_id / "traces.jsonl",
    ]
    for path in paths:
        if path.exists():
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []


def _action_mix(summary: dict[str, Any]) -> dict[str, Any]:
    counts = summary.get("status_counts") if isinstance(summary.get("status_counts"), dict) else {}
    return {
        "abstain": counts.get("abstain", 0),
        "watch": summary.get("watch_count", counts.get("watch", 0)),
        "bid": summary.get("bid_action_count", 0),
        "status_counts": counts,
    }


def _shadow_metrics(run_id: str) -> dict[str, Any] | None:
    shadows = []
    for row in _traces(run_id):
        for call in row.get("tool_calls") or []:
            if call.get("name") == "shadow_required_simulation":
                shadows.append(call.get("result") or {})
    if not shadows:
        return None
    rejected = [result for result in shadows if result.get("shadow_accepted") is not True]
    negative_worst = [
        result
        for result in shadows
        if _float_or_none(result.get("shadow_worst_case_profit_eur")) is not None
        and _float_or_none(result.get("shadow_worst_case_profit_eur")) < 0
    ]
    return {
        "shadow_bid_count": len(shadows),
        "would_have_been_blocked_count": len(rejected),
        "shadow_reject_rate": round(len(rejected) / len(shadows), 6) if shadows else None,
        "shadow_negative_worst_case_count": len(negative_worst),
        "shadow_negative_worst_case_rate": round(len(negative_worst) / len(shadows), 6) if shadows else None,
    }


def _context_only_tool_leak_count(run_id: str) -> int:
    hidden = {
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
    leaks = 0
    for row in _traces(run_id):
        for call in row.get("tool_calls") or []:
            name = str(call.get("name") or "")
            if name == "shadow_required_simulation":
                continue
            if name in hidden or (name.startswith("get_") and name.endswith("_bid_feasibility")):
                leaks += 1
    return leaks


def _shadow_rejection_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row_meta in rows:
        run_id = row_meta["run_id"]
        for row in _traces(run_id):
            decision = row.get("decision") or {}
            if decision.get("action") != "bid":
                continue
            for call in row.get("tool_calls") or []:
                if call.get("name") != "shadow_required_simulation":
                    continue
                result = call.get("result") or {}
                if result.get("shadow_accepted") is True:
                    continue
                examples.append(
                    {
                        "run_id": run_id,
                        "step": row.get("step"),
                        "timestamp": row.get("timestamp"),
                        "agent_id": row.get("agent_id"),
                        "archetype": row.get("archetype"),
                        "decision": decision,
                        "shadow_reason_codes": result.get("shadow_reason_codes", []),
                        "shadow_worst_case_profit_eur": result.get("shadow_worst_case_profit_eur"),
                    }
                )
    return examples


def _delta(left: Any, right: Any) -> float | None:
    try:
        return round(float(left) - float(right), 6)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
