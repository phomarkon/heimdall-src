"""Forecaster leaderboard, baseline parsing, and forecaster summary helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from heimdall_run_view._utils import (
    RunContext,
    _float,
    _optional_float,
    _optional_int,
    _read_json,
    default_repo_root,
)


def _forecaster_leaderboard() -> list[dict[str, Any]]:
    rows = _parse_forecaster_leaderboard(
        default_repo_root() / "notes" / "forecaster_leaderboard.md"
    )
    seen = {str(row.get("model_id", "")).lower() for row in rows}
    for model_id, label, available in (
        ("ar1", "Gaussian AR(1) fallback", True),
        ("f9", "TimesFM-2.0 zero-shot", importlib.util.find_spec("timesfm") is not None),
        ("f10", "Chronos-Bolt zero-shot", importlib.util.find_spec("chronos") is not None),
        ("f11", "PriceFM-shaped PatchTST surrogate", True),
    ):
        if model_id in seen:
            continue
        rows.append(
            {
                "model_id": model_id,
                "label": label,
                "seed_count": None,
                "q10_pinball": None,
                "q50_pinball": None,
                "q90_pinball": None,
                "mean_pinball": None,
                "raw_coverage": None,
                "aci_coverage": None,
                "status": "registered" if available else "registered / dependency missing",
            }
        )
    return rows


def _parse_forecaster_leaderboard(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("|-") or "Model" in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 8:
            continue
        rows.append(
            {
                "model_id": cells[0],
                "label": _forecaster_label(cells[0]),
                "seed_count": _int_cell(cells[1]),
                "q10_pinball": cells[2],
                "q50_pinball": cells[3],
                "q90_pinball": cells[4],
                "mean_pinball": cells[5],
                "raw_coverage": cells[6],
                "aci_coverage": cells[7],
                "status": _forecaster_status(cells[0]),
            }
        )
    return rows


def _forecaster_label(model_id: str) -> str:
    labels = {
        "f0": "Seasonal AR(24)",
        "f1_lgbm": "LightGBM quantile",
        "f2_blr": "Bayesian linear regression",
        "f3": "DeepAR-Lite",
        "f3_ensemble": "5-seed PatchTST ensemble",
        "f4_mc_dropout": "PatchTST MC dropout",
        "f7": "PatchTST split-CP",
        "f8": "Multivariate PatchTST ACI",
        "b1_random_walk": "Random walk baseline",
        "b2_ewma": "EWMA baseline",
        "b3_seasonal_naive": "Seasonal naive baseline",
        "b4_lightgbm_quantile": "LightGBM baseline",
        "b7_nbeats_lite": "N-BEATS lite baseline",
    }
    return labels.get(model_id, model_id)


def _forecaster_status(model_id: str) -> str:
    model = model_id.lower()
    if model.startswith("b"):
        return "validation baseline"
    if model in {
        "f0",
        "f1_lgbm",
        "f2_blr",
        "f3",
        "f3_ensemble",
        "f4_mc_dropout",
        "f7",
        "f8",
        "f11",
    }:
        return "usable"
    return "recorded"


def _int_cell(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _focal_baselines() -> list[dict[str, Any]]:
    """Build the focal-policy baseline leaderboard strictly from evaluated runs.

    Only evaluations that actually exist on disk are reported; nothing is
    fabricated. Single-run ``run_summary.json`` baseline dirs become one row
    each, and the verifier-ablation ``paired_summary.json`` contributes one
    aggregate row per arm.
    """
    root = default_repo_root()
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("research/llm/evaluations/*baseline*/run_summary.json")):
        data = _read_json(summary_path)
        if not data:
            continue
        rows.append(_focal_baseline_row(data, summary_path.parent.name))
    for paired_path in sorted(root.glob("research/llm/evaluations/*verifierless*/paired_summary.json")):
        rows.extend(_verifierless_baseline_rows(_read_json(paired_path), paired_path.parent.name))
    return rows


def _focal_baseline_row(data: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": str(data.get("run_id") or run_id),
        "label": _focal_baseline_label(run_id),
        "kind": "baseline",
        "profit_eur": _optional_float(data.get("cumulative_pnl_eur")),
        "realized_profit_eur": _optional_float(data.get("realized_profit_eur")),
        "cvar_95_eur": _optional_float(data.get("downside_cvar_95_eur")),
        "fill_rate": _optional_float(data.get("fill_rate")),
        "bid_count": _optional_int(data.get("bid_count")),
        "regret_eur": _optional_float(data.get("regret_eur")),
        "n_runs": 1,
        "status": "evaluated",
        "source": f"evaluations/{run_id}/run_summary.json",
    }


def _verifierless_baseline_rows(data: dict[str, Any], matrix_id: str) -> list[dict[str, Any]]:
    paired = data.get("rows") or []
    out: list[dict[str, Any]] = []

    def _arm_values(metric_key: str, arm: str) -> list[float]:
        values: list[float] = []
        for row in paired:
            metric = (row.get("metrics") or {}).get(metric_key)
            value = metric.get(arm) if isinstance(metric, dict) else None
            if isinstance(value, int | float):
                values.append(float(value))
        return values

    for arm in ("guarded", "variant"):
        profits = _arm_values("realized_profit_eur", arm)
        cvars = _arm_values("downside_cvar_95_eur", arm)
        if not profits:
            continue
        out.append(
            {
                "run_id": f"{matrix_id}:{arm}",
                "label": f"Verifier ablation — {arm}",
                "kind": "ablation",
                "profit_eur": None,
                "realized_profit_eur": round(sum(profits) / len(profits), 2),
                "cvar_95_eur": round(sum(cvars) / len(cvars), 2) if cvars else None,
                "fill_rate": None,
                "bid_count": None,
                "regret_eur": None,
                "n_runs": len(profits),
                "status": "ablation mean",
                "source": f"evaluations/{matrix_id}/paired_summary.json",
            }
        )
    return out


def _focal_baseline_label(run_id: str) -> str:
    lowered = run_id.lower()
    if "profitguard" in lowered:
        suffix = run_id.split("profitguard-", 1)[-1] if "profitguard-" in lowered else ""
        return f"Profit-guard baseline {suffix}".strip()
    if "verifierless" in lowered:
        return "Verifier ablation"
    return run_id.replace("-", " ")


def _forecaster_summary(
    *,
    run_id: str,
    total_steps: int,
    rows_by_step: dict[int, list[dict[str, Any]]],
    snapshots: list[dict[str, Any]],
    context: RunContext,
) -> dict[str, Any]:
    from heimdall_run_view._priority import _priority_accuracy, _priority_signals

    counts: dict[str, int] = {}
    run_ids_by_forecaster: dict[str, list[str]] = {}
    repo_root = default_repo_root()
    for rows in rows_by_step.values():
        for row in rows:
            forecaster = _normalize_forecaster_id(row.get("forecaster_id"))
            counts[forecaster] = counts.get(forecaster, 0) + 1
            run_ids_by_forecaster.setdefault(forecaster, [run_id])
    for path in repo_root.glob("research/llm/evaluations/*/run_summary.json"):
        forecaster = _forecaster_from_run_id(path.parent.name)
        if forecaster:
            bucket = run_ids_by_forecaster.setdefault(forecaster, [])
            if path.parent.name not in bucket and len(bucket) < 8:
                bucket.append(path.parent.name)
    active = max(counts.items(), key=lambda item: item[1])[0] if counts else "unavailable"
    final = snapshots[-1] if snapshots else {}
    priority = (
        _priority_accuracy(_priority_signals(total_steps, rows_by_step, context), context)
        if total_steps
        else {}
    )
    return {
        "active_forecaster_id": active,
        "run_ids_by_forecaster": run_ids_by_forecaster,
        "coverage": (final.get("health") or {}).get("coverage", 0.0),
        "accepted_bid_rate": (final.get("health") or {}).get("verifier_acceptance_rate", 0.0),
        "cumulative_pnl_eur": (final.get("health") or {}).get("cumulative_pnl_eur", 0.0),
        "selected_tick_count": priority.get("selected_tick_count", 0),
    }


def _forecaster_from_run_id(run_id: str) -> str | None:
    for token in ("f3_ensemble", "f11", "f10", "f9", "f8", "f7", "f0", "ar1"):
        if f"-{token}" in run_id.lower() or run_id.lower().endswith(token):
            return token
    return None


def _normalize_forecaster_id(value: Any) -> str:
    text = str(value or "unavailable").strip()
    return text.lower() if text.upper().startswith("F") else text


def _forecast_diagnostics(
    row: dict[str, Any],
    market: dict[str, Any],
    selected_trace: dict[str, Any],
) -> dict[str, Any]:
    interval = row.get("forecast_interval_eur_mwh") or []
    low = _float(interval[0], market["mfrr_price_eur_per_mwh"]) if len(interval) > 0 else None
    high = _float(interval[1], market["mfrr_price_eur_per_mwh"]) if len(interval) > 1 else None
    realized = _float(row.get("market_price_eur_mwh"), market["mfrr_price_eur_per_mwh"])
    signals = _forecast_signals(row)
    worst_case = (selected_trace.get("verifier_verdict") or {}).get("worst_case_profit_eur")
    return {
        "forecaster_id": _normalize_forecaster_id(row.get("forecaster_id")),
        "interval_low_eur_mwh": low,
        "interval_high_eur_mwh": high,
        "interval_width_eur_mwh": round(high - low, 6)
        if low is not None and high is not None
        else None,
        "realized_price_eur_mwh": realized,
        "covered": bool(low <= realized <= high) if low is not None and high is not None else None,
        "spot_mfrr_spread_eur_mwh": round(
            market["mfrr_price_eur_per_mwh"] - market["dk1_price_eur_per_mwh"], 6
        ),
        "up_edge_eur_mwh": signals.get("up_edge_lower_minus_spot_eur_mwh"),
        "down_edge_eur_mwh": signals.get("down_edge_spot_minus_upper_eur_mwh"),
        "expected_spread_eur_mwh": signals.get("expected_spread_eur_mwh"),
        "worst_case_profit_eur": worst_case,
    }


def _forecast_signals(row: dict[str, Any]) -> dict[str, float]:
    signals: dict[str, float] = {}
    for call in row.get("tool_calls") or []:
        result = call.get("result") or {}
        nested = result.get("signals") or {}
        for key in (
            "up_edge_lower_minus_spot_eur_mwh",
            "down_edge_spot_minus_upper_eur_mwh",
            "expected_spread_eur_mwh",
            "recent_up_spread_eur_mwh",
        ):
            value = result.get(key, nested.get(key))
            if isinstance(value, int | float):
                signals[key] = round(float(value), 6)
    return signals
