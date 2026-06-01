from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

KS = (10, 12, 24)
PRIORITY_LABEL_WEIGHT = {"low": 0.1, "medium": 0.45, "high": 0.75, "critical": 1.0}
WATCH_WEIGHT = {"ignore": 0.0, "watch": 0.45, "must_watch": 1.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate priority-calibration rankers against ex-post MTU opportunity.")
    parser.add_argument("--config-list", type=Path, help="Config list used for the run matrix.")
    parser.add_argument("--run-id", action="append", default=[], help="Run id to evaluate; may be repeated.")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations/priority-calibration-matrix"))
    args = parser.parse_args()

    run_ids = list(args.run_id)
    if args.config_list:
        run_ids.extend(_run_ids_from_config_list(args.config_list))
    run_ids = sorted(dict.fromkeys(run_ids))
    if not run_ids:
        raise SystemExit("provide --config-list or at least one --run-id")

    result = evaluate_priority_calibration(run_ids, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


def evaluate_priority_calibration(run_ids: list[str], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for run_id in run_ids:
        try:
            traces, truth, bids, summary = _load_run_inputs(run_id)
        except FileNotFoundError:
            missing.append(run_id)
            continue
        if traces.empty or "run_id" not in traces.columns or truth.empty:
            missing.append(run_id)
            continue
        ticks = _tick_features(traces, truth, bids)
        if ticks.empty:
            continue
        scores = _ranker_scores(ticks)
        for method, values in scores.items():
            for k in KS:
                metric_rows.append(_metrics_for_k(run_id, method, k, ticks, values))
        quality_rows.append(_quality_row(run_id, traces, ticks, summary))

    metrics = pd.DataFrame(metric_rows)
    quality = pd.DataFrame(quality_rows)
    metrics_path = output_dir / "priority_ranker_metrics.csv"
    quality_path = output_dir / "priority_run_quality.csv"
    report_path = output_dir / "priority_calibration_report.md"
    metrics.to_csv(metrics_path, index=False)
    quality.to_csv(quality_path, index=False)
    _write_report(report_path, metrics, quality, missing)
    return {
        "ok": True,
        "run_count": len(run_ids) - len(missing),
        "missing_runs": missing,
        "metrics": str(metrics_path),
        "quality": str(quality_path),
        "report": str(report_path),
    }


def _run_ids_from_config_list(path: Path) -> list[str]:
    run_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = yaml.safe_load(Path(line.strip()).read_text(encoding="utf-8")) or {}
        run_id = payload.get("run_id")
        if run_id:
            run_ids.append(str(run_id))
    return run_ids


def _load_run_inputs(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eval_dir = Path("evaluations") / run_id
    manifest_path = eval_dir / "manifest.json"
    summary_path = eval_dir / "run_summary.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = Path(manifest["run_dir"])
    truth_dir = Path(manifest["truth_dir"])
    traces = _load_traces(run_dir / "traces.jsonl")
    truth = _load_truth(truth_dir / "activation_truth.parquet")
    bids_path = eval_dir / "bid_evaluations.parquet"
    bids = pd.read_parquet(bids_path) if bids_path.exists() else pd.DataFrame()
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return traces, truth, bids, summary


def _load_traces(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        decision = payload.get("decision") or {}
        rows.append(
            {
                "run_id": payload.get("run_id"),
                "step": int(payload.get("step", 0)),
                "timestamp_utc": pd.to_datetime(payload.get("timestamp"), utc=True),
                "zone": payload.get("zone") or "DK1",
                "agent_id": payload.get("agent_id"),
                "action": decision.get("action"),
                "side": decision.get("side"),
                "confidence": _num(decision.get("confidence")),
                "watch_label": decision.get("watch_label", "ignore"),
                "priority_label": decision.get("priority_label"),
                "priority_score": _num(decision.get("priority_score")),
                "operator_action": decision.get("operator_action"),
                "priority_reason": decision.get("priority_reason"),
                "verifier_accepted": payload.get("verifier_accepted"),
                "verifier_reason_codes": payload.get("verifier_reason_codes") or [],
                "tool_calls": payload.get("tool_calls") or [],
                "rationale": payload.get("rationale") or decision.get("rationale") or "",
            }
        )
    return pd.DataFrame(rows)


def _load_truth(path: Path) -> pd.DataFrame:
    truth = pd.read_parquet(path)
    if "timestamp_utc" not in truth.columns:
        for candidate in ("timestamp", "delivery_start_utc", "utc_timestamp"):
            if candidate in truth.columns:
                truth = truth.rename(columns={candidate: "timestamp_utc"})
                break
    truth["timestamp_utc"] = pd.to_datetime(truth["timestamp_utc"], utc=True)
    return truth


def _tick_features(traces: pd.DataFrame, truth: pd.DataFrame, bids: pd.DataFrame) -> pd.DataFrame:
    truth_tick = _truth_importance_by_tick(truth)
    bid_tick = _realized_positive_by_tick(bids)
    rows: list[dict[str, Any]] = []
    for (run_id, step, tick_timestamp, zone), group in traces.groupby(["run_id", "step", "timestamp_utc", "zone"], sort=True):
        tool_stats = _tool_stats(group["tool_calls"])
        watch_values = group["watch_label"].map(lambda value: WATCH_WEIGHT.get(str(value), 0.0))
        priority_values = group["priority_score"].fillna(0.0).clip(0.0, 1.0)
        label_values = group["priority_label"].map(lambda value: PRIORITY_LABEL_WEIGHT.get(str(value), 0.0))
        sides = [side for side in group["side"].dropna().astype(str).tolist() if side in {"up", "down"}]
        side_consensus = max((sides.count(side) for side in ("up", "down")), default=0) / max(len(sides), 1)
        timestamp = pd.Timestamp(tick_timestamp)
        rows.append(
            {
                "run_id": run_id,
                "step": step,
                "timestamp_utc": timestamp,
                "zone": zone,
                "agent_count": len(group),
                "must_watch_share": float((group["watch_label"] == "must_watch").mean()),
                "watch_share": float(group["watch_label"].isin(["watch", "must_watch"]).mean()),
                "mean_watch_score": float(watch_values.mean()),
                "mean_priority_score": float(priority_values.mean()),
                "max_priority_score": float(priority_values.max()),
                "mean_priority_label_score": float(label_values.mean()),
                "critical_share": float((group["priority_label"] == "critical").mean()),
                "high_or_critical_share": float(group["priority_label"].isin(["high", "critical"]).mean()),
                "accepted_candidate_count": tool_stats["accepted_candidate_count"],
                "accepted_agent_share": tool_stats["accepted_agent_share"],
                "side_consensus": float(side_consensus),
                "max_expected_profit_eur": tool_stats["max_expected_profit_eur"],
                "max_worst_case_profit_eur": tool_stats["max_worst_case_profit_eur"],
                "mean_tool_watch_score": tool_stats["mean_tool_watch_score"],
                "rejection_risk": tool_stats["rejection_risk"],
                "truth_importance": float(truth_tick.get((timestamp, zone), 0.0)),
                "realized_positive_profit_eur": float(bid_tick.get((timestamp, zone), 0.0)),
            }
        )
    frame = pd.DataFrame(rows)
    for column in ("max_expected_profit_eur", "max_worst_case_profit_eur", "accepted_candidate_count"):
        frame[f"norm_{column}"] = _normalize(frame[column])
    return frame


def _truth_importance_by_tick(truth: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], float]:
    side_col = _first_existing(truth, ["side", "activation_side", "activation_direction", "direction"])
    volume_col = _first_existing(truth, ["activated_volume_mwh", "activation_volume_mwh", "volume_mwh"])
    price_col = _first_existing(truth, ["settlement_price_eur_mwh", "settlement_price", "imbalance_price_eur_mwh"])
    spot_col = _first_existing(truth, ["spot_price_eur_mwh", "day_ahead_price_eur_mwh", "market_price_eur_mwh"])
    if side_col is None or volume_col is None or price_col is None or spot_col is None:
        return {}

    scores: dict[tuple[pd.Timestamp, str], float] = {}
    for _, row in truth.iterrows():
        side = str(row.get(side_col, "")).lower()
        settlement = _num(row.get(price_col))
        spot = _num(row.get(spot_col))
        volume = max(0.0, _num(row.get(volume_col)))
        if side == "up":
            profit_per_mwh = settlement - spot
        elif side == "down":
            profit_per_mwh = spot - settlement
        else:
            profit_per_mwh = 0.0
        timestamp = pd.Timestamp(row["timestamp_utc"])
        key = (timestamp, str(row.get("zone", "DK1")))
        scores[key] = scores.get(key, 0.0) + max(0.0, profit_per_mwh) * volume
    return scores


def _realized_positive_by_tick(bids: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], float]:
    if bids.empty or "timestamp_utc" not in bids.columns or "realized_profit_eur" not in bids.columns:
        return {}
    frame = bids.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    if "zone" not in frame.columns:
        frame["zone"] = "DK1"
    frame["positive_profit"] = frame["realized_profit_eur"].clip(lower=0.0)
    return frame.groupby(["timestamp_utc", "zone"])["positive_profit"].sum().to_dict()


def _tool_stats(tool_groups: pd.Series) -> dict[str, float]:
    accepted = 0
    agents_with_accepted = 0
    agent_count = len(tool_groups)
    expected: list[float] = []
    worst: list[float] = []
    watch_scores: list[float] = []
    rejections = 0
    simulations = 0
    for calls in tool_groups:
        agent_accepted = False
        for call in calls or []:
            result = call.get("result") or {}
            name = str(call.get("name", ""))
            if "watch_score" in result:
                watch_scores.append(_num(result.get("watch_score")))
            if not name.startswith("simulate"):
                continue
            simulations += 1
            if result.get("accepted") is True:
                accepted += 1
                agent_accepted = True
            else:
                rejections += 1
            expected.append(_num(result.get("rough_expected_profit_eur")))
            worst.append(_num(result.get("worst_case_profit_eur")))
        if agent_accepted:
            agents_with_accepted += 1
    return {
        "accepted_candidate_count": float(accepted),
        "accepted_agent_share": agents_with_accepted / max(agent_count, 1),
        "max_expected_profit_eur": max(expected, default=0.0),
        "max_worst_case_profit_eur": max(worst, default=0.0),
        "mean_tool_watch_score": float(np.mean(watch_scores)) if watch_scores else 0.0,
        "rejection_risk": rejections / max(simulations, 1),
    }


def _ranker_scores(ticks: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        # --- single-feature probes: isolate what actually carries the signal ---
        "sim_worst_case_only": ticks["norm_max_worst_case_profit_eur"],
        "sim_expected_only": ticks["norm_max_expected_profit_eur"],
        "accepted_only": ticks["accepted_agent_share"],
        "side_consensus_only": ticks["side_consensus"],
        # LLM self-reported only — D4 control; expected ~random (anti-informative)
        "llm_self_report_only": (
            0.5 * ticks["mean_priority_score"]
            + 0.3 * ticks["watch_share"]
            + 0.2 * ticks["mean_watch_score"]
        ),
        # --- clean simulator-grounded blend (NO llm watch/priority fields) ---
        "sim_grounded": (
            0.60 * ticks["norm_max_worst_case_profit_eur"]
            + 0.40 * ticks["norm_max_expected_profit_eur"]
            + 0.45 * ticks["accepted_agent_share"]
            + 0.20 * ticks["side_consensus"]
            - 0.25 * ticks["rejection_risk"]
        ),
        # --- existing hand-weighted candidates (mix in LLM watch/priority) ---
        "agent_priority_only": (
            0.55 * ticks["mean_priority_score"]
            + 0.30 * ticks["max_priority_score"]
            + 0.15 * ticks["mean_priority_label_score"]
        ),
        "trace_hybrid": (
            0.45 * ticks["must_watch_share"]
            + 0.30 * ticks["watch_share"]
            + 0.35 * ticks["high_or_critical_share"]
            + 0.55 * ticks["accepted_agent_share"]
            + 0.30 * ticks["side_consensus"]
            + 0.50 * ticks["norm_max_expected_profit_eur"]
            + 0.65 * ticks["norm_max_worst_case_profit_eur"]
            + 0.25 * ticks["mean_tool_watch_score"]
            - 0.25 * ticks["rejection_risk"]
        ),
        "budgeted_chair": (
            0.20 * ticks["mean_priority_score"]
            + 0.20 * ticks["must_watch_share"]
            + 0.15 * ticks["critical_share"]
            + 0.50 * ticks["accepted_agent_share"]
            + 0.25 * ticks["side_consensus"]
            + 0.20 * ticks["norm_max_expected_profit_eur"]
            + 0.70 * ticks["norm_max_worst_case_profit_eur"]
            - 0.35 * ticks["rejection_risk"]
        ),
        "frontend_baseline": (
            0.55 * ticks["mean_watch_score"]
            + 0.35 * ticks["watch_share"]
            + 0.45 * ticks["accepted_agent_share"]
            + 0.40 * ticks["side_consensus"]
            + 0.45 * ticks["norm_max_expected_profit_eur"]
            + 0.45 * ticks["norm_max_worst_case_profit_eur"]
            + 0.25 * ticks["mean_tool_watch_score"]
        ),
    }


def _metrics_for_k(run_id: str, method: str, k: int, ticks: pd.DataFrame, score: pd.Series) -> dict[str, Any]:
    frame = ticks.assign(score=score.fillna(0.0)).sort_values(["score", "timestamp_utc"], ascending=[False, True])
    selected = frame.head(min(k, len(frame)))
    truth_total = max(float(frame["truth_importance"].sum()), 0.0)
    selected_truth = float(selected["truth_importance"].sum())
    true_top = set(frame.sort_values(["truth_importance", "timestamp_utc"], ascending=[False, True]).head(min(k, len(frame)))["timestamp_utc"])
    selected_set = set(selected["timestamp_utc"])
    random_capture = min(k, len(frame)) / max(len(frame), 1)
    realized_total = max(float(frame["realized_positive_profit_eur"].sum()), 0.0)
    return {
        "run_id": run_id,
        "method": method,
        "k": k,
        "top_k_oracle_capture": _safe_div(selected_truth, truth_total),
        "hit_at_k": _safe_div(len(selected_set & true_top), max(len(true_top), 1)),
        "lift_over_random": _safe_div(_safe_div(selected_truth, truth_total), random_capture),
        "ndcg_at_k": _ndcg(frame, selected, k),
        "realized_positive_profit_capture": _safe_div(float(selected["realized_positive_profit_eur"].sum()), realized_total),
        "selected_truth_importance": selected_truth,
        "truth_importance_total": truth_total,
        "selected_ticks": len(selected),
        "available_ticks": len(frame),
    }


def _quality_row(run_id: str, traces: pd.DataFrame, ticks: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    priority_present = traces[["priority_label", "priority_score", "operator_action", "priority_reason"]].notna().all(axis=1)
    fallback_failures = traces["rationale"].astype(str).str.contains("LLM call failed", case=False, na=False).sum()
    expected_rows = int(summary.get("trace_rows_expected") or summary.get("expected_trace_rows") or len(traces))
    return {
        "run_id": run_id,
        "trace_rows": len(traces),
        "expected_trace_rows": expected_rows,
        "trace_row_complete": len(traces) == expected_rows,
        "priority_field_coverage": float(priority_present.mean()) if len(traces) else 0.0,
        "llm_fallback_failures": int(fallback_failures),
        "verifier_realized_profit_breach_rate": summary.get("verifier_realized_profit_breach_rate"),
        "evidence_reference_validity": summary.get("evidence_reference_validity"),
        "truth_positive_ticks": int((ticks["truth_importance"] > 0).sum()),
        "truth_importance_total": float(ticks["truth_importance"].sum()),
    }


def _write_report(path: Path, metrics: pd.DataFrame, quality: pd.DataFrame, missing: list[str]) -> None:
    lines = ["# Priority Calibration Report", ""]
    if missing:
        lines.extend(["Missing evaluated runs:", *[f"- {run_id}" for run_id in missing], ""])
    if not metrics.empty:
        average = (
            metrics.groupby(["method", "k"], as_index=False)[
                ["top_k_oracle_capture", "hit_at_k", "lift_over_random", "ndcg_at_k", "realized_positive_profit_capture"]
            ]
            .mean()
            .sort_values(["k", "top_k_oracle_capture"], ascending=[True, False])
        )
        lines.append("## Mean Metrics")
        lines.append("")
        lines.append(average.to_markdown(index=False, floatfmt=".3f"))
        lines.append("")
    if not quality.empty:
        lines.append("## Run Quality")
        lines.append("")
        lines.append(quality.to_markdown(index=False, floatfmt=".3f"))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ndcg(frame: pd.DataFrame, selected: pd.DataFrame, k: int) -> float:
    gains = selected["truth_importance"].head(k).to_numpy(dtype=float)
    discounts = np.array([1.0 / math.log2(i + 2) for i in range(len(gains))])
    dcg = float(np.sum(gains * discounts))
    ideal = frame.sort_values("truth_importance", ascending=False)["truth_importance"].head(k).to_numpy(dtype=float)
    ideal_discounts = np.array([1.0 / math.log2(i + 2) for i in range(len(ideal))])
    idcg = float(np.sum(ideal * ideal_discounts))
    return _safe_div(dcg, idcg)


def _normalize(series: pd.Series) -> pd.Series:
    values = series.fillna(0.0).clip(lower=0.0)
    maximum = float(values.max()) if len(values) else 0.0
    if maximum <= 0.0:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values / maximum).clip(0.0, 1.0)


def _first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _num(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


if __name__ == "__main__":
    main()
