from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from packages.data import file_sha256

BID_COLUMNS = [
    "run_id",
    "step",
    "timestamp_utc",
    "agent_id",
    "archetype",
    "zone",
    "side",
    "quantity_mwh",
    "limit_price_eur_mwh",
    "verifier_accepted",
    "status",
    "cleared_mwh",
    "activated_volume_mwh",
    "realized_profit_eur",
    "profit_per_mwh",
    "price_distance_eur_mwh",
    "forecast_interval_covered",
]


@dataclass(frozen=True)
class EvaluationInputs:
    run_dir: Path
    context_dir: Path
    truth_dir: Path
    output_dir: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate completed AI-society bids against ex-post truth.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Defaults to evaluations/<run_id>.")
    args = parser.parse_args()

    run_id = args.run_dir.name
    output_dir = args.output_dir or Path("evaluations") / run_id
    result = evaluate_society_run(
        EvaluationInputs(
            run_dir=args.run_dir,
            context_dir=args.context_dir,
            truth_dir=args.truth_dir,
            output_dir=output_dir,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def evaluate_society_run(inputs: EvaluationInputs) -> dict[str, Any]:
    context_manifest_path = inputs.context_dir / "manifest.json"
    truth_manifest_path = inputs.truth_dir / "truth_manifest.json"
    context_manifest = _read_manifest(context_manifest_path, expected_visibility="agent_context")
    truth_manifest = _read_manifest(truth_manifest_path, expected_visibility="evaluation_only")

    traces_path = inputs.run_dir / "traces.jsonl"
    truth_path = inputs.truth_dir / "activation_truth.parquet"
    if not traces_path.exists():
        raise FileNotFoundError(f"society run trace file missing: {traces_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"evaluation truth file missing: {truth_path}")

    traces = _load_traces(traces_path)
    truth = _load_truth(truth_path)
    bids = _score_bids(traces, truth)
    agent_metrics = _agent_metrics(bids)
    archetype_metrics = _archetype_metrics(bids)
    run_summary = _run_summary(bids, traces, truth)

    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    bid_path = inputs.output_dir / "bid_evaluations.parquet"
    agent_path = inputs.output_dir / "agent_metrics.parquet"
    archetype_path = inputs.output_dir / "archetype_metrics.parquet"
    summary_path = inputs.output_dir / "run_summary.json"
    manifest_path = inputs.output_dir / "manifest.json"

    bids.to_parquet(bid_path, index=False)
    agent_metrics.to_parquet(agent_path, index=False)
    archetype_metrics.to_parquet(archetype_path, index=False)
    summary_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "created_at_utc": _iso_z(datetime.now(tz=UTC)),
        "schema_version": "1.0.0",
        "run_id": run_summary["run_id"],
        "run_dir": str(inputs.run_dir),
        "context_dir": str(inputs.context_dir),
        "truth_dir": str(inputs.truth_dir),
        "trace_sha256": file_sha256(traces_path),
        "context_manifest_sha256": file_sha256(context_manifest_path),
        "truth_manifest_sha256": file_sha256(truth_manifest_path),
        "context_dataset_id": context_manifest.get("dataset_id"),
        "truth_dataset_id": truth_manifest.get("dataset_id"),
        "row_counts": {
            "traces": len(traces),
            "bid_evaluations": len(bids),
            "agent_metrics": len(agent_metrics),
            "archetype_metrics": len(archetype_metrics),
            "truth": len(truth),
        },
        "artifacts": {
            "bid_evaluations": str(bid_path),
            "agent_metrics": str(agent_path),
            "archetype_metrics": str(archetype_path),
            "run_summary": str(summary_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"ok": True, "output_dir": str(inputs.output_dir), "run_summary": run_summary}


def _score_bids(traces: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, group in traces.sort_values(["timestamp_utc", "agent_id"]).groupby(["timestamp_utc", "zone"], sort=False):
        timestamp = pd.Timestamp(group["timestamp_utc"].iloc[0])
        zone = str(group["zone"].iloc[0])
        truth_rows = truth[(truth["timestamp_utc"] == timestamp) & (truth["zone"] == zone)]
        rows.extend(_score_tick(group, truth_rows))
    if not rows:
        return pd.DataFrame(columns=BID_COLUMNS)
    return pd.DataFrame(rows, columns=BID_COLUMNS)


def _score_tick(traces: pd.DataFrame, truth_rows: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    bids = []
    for _, trace in traces.iterrows():
        base = _base_row(trace)
        invalid = _invalid_reason(trace)
        if invalid is not None:
            out.append({**base, "status": invalid})
            continue
        bids.append((trace, base))

    for side in ["up", "down"]:
        side_truth = _truth_for_side(truth_rows, side)
        side_bids = [(trace, base) for trace, base in bids if trace["side"] == side]
        if side_truth is None:
            for _, base in side_bids:
                out.append({**base, "status": "missing_truth"})
            continue
        direction = str(side_truth["activation_direction"])
        if direction != side:
            for _, base in side_bids:
                out.append({**base, "status": "wrong_side" if direction != "neutral" else "no_activation"})
            continue
        settlement_price = float(side_truth["settlement_price_eur_mwh"])
        spot_price = float(side_truth["spot_price_eur_mwh"])
        activated_volume = max(0.0, float(side_truth["activated_volume_mwh"]))
        remaining = activated_volume
        for trace, base in sorted(side_bids, key=lambda item: (float(item[0]["limit_price_eur_mwh"]), str(item[0]["agent_id"]))):
            limit_price = float(trace["limit_price_eur_mwh"])
            if limit_price - settlement_price > 1e-9:
                profit_per_mwh = _profit_per_mwh(side=side, spot_price=spot_price, clearing_price=settlement_price)
                out.append(
                    {
                        **base,
                        "status": "price_not_crossed",
                        "activated_volume_mwh": activated_volume,
                        "profit_per_mwh": round(profit_per_mwh, 6),
                        "price_distance_eur_mwh": round(limit_price - settlement_price, 6),
                        "forecast_interval_covered": _interval_covered(trace, settlement_price),
                    }
                )
                continue
            cleared = min(float(trace["quantity_mwh"]), remaining)
            if cleared <= 1e-9:
                out.append({**base, "status": "no_activation", "activated_volume_mwh": activated_volume})
                continue
            remaining = round(remaining - cleared, 12)
            profit = _settlement(side=side, quantity_mwh=cleared, spot_price=spot_price, clearing_price=settlement_price)
            out.append(
                {
                    **base,
                    "status": "filled" if cleared + 1e-9 >= float(trace["quantity_mwh"]) else "partially_filled",
                    "cleared_mwh": round(cleared, 6),
                    "activated_volume_mwh": activated_volume,
                    "realized_profit_eur": round(profit, 6),
                    "profit_per_mwh": round(profit / cleared, 6) if cleared > 0 else None,
                    "price_distance_eur_mwh": round(limit_price - settlement_price, 6),
                    "forecast_interval_covered": _interval_covered(trace, settlement_price),
                }
            )

    trace_ids = {(row["step"], row["agent_id"]) for row in out}
    for _trace, base in bids:
        if (base["step"], base["agent_id"]) not in trace_ids:
            out.append({**base, "status": "wrong_side"})
    return out


def _base_row(trace: pd.Series) -> dict[str, Any]:
    return {
        "run_id": str(trace["run_id"]),
        "step": int(trace["step"]),
        "timestamp_utc": pd.Timestamp(trace["timestamp_utc"]),
        "agent_id": str(trace["agent_id"]),
        "archetype": str(trace["archetype"]),
        "zone": str(trace["zone"]),
        "side": trace["side"],
        "quantity_mwh": trace["quantity_mwh"],
        "limit_price_eur_mwh": trace["limit_price_eur_mwh"],
        "verifier_accepted": trace["verifier_accepted"],
        "status": "eligible",
        "cleared_mwh": 0.0,
        "activated_volume_mwh": 0.0,
        "realized_profit_eur": 0.0,
        "profit_per_mwh": None,
        "price_distance_eur_mwh": None,
        "forecast_interval_covered": None,
    }


def _invalid_reason(trace: pd.Series) -> str | None:
    if trace["action"] in {"abstain", "watch"}:
        return str(trace["action"])
    submitted_at = trace.get("submitted_at_utc")
    if isinstance(submitted_at, pd.Timestamp) and submitted_at > pd.Timestamp(trace["timestamp_utc"]) - pd.Timedelta(minutes=45):
        return "gate_closed"
    for column in ["side", "quantity_mwh", "limit_price_eur_mwh"]:
        if pd.isna(trace[column]):
            return "invalid"
    if float(trace["quantity_mwh"]) <= 0:
        return "invalid"
    if trace["side"] not in {"up", "down"}:
        return "invalid"
    return None


def _truth_for_side(truth_rows: pd.DataFrame, side: str) -> pd.Series | None:
    if truth_rows.empty:
        return None
    matches = truth_rows[truth_rows["activation_direction"] == side]
    if matches.empty:
        neutral = truth_rows[truth_rows["activation_direction"] == "neutral"]
        return neutral.iloc[0] if not neutral.empty else truth_rows.iloc[0]
    return matches.sort_values("activated_volume_mwh", ascending=False).iloc[0]


def _agent_metrics(bids: pd.DataFrame) -> pd.DataFrame:
    if bids.empty:
        return pd.DataFrame()
    rows = []
    for agent_id, group in bids.groupby("agent_id"):
        rows.append(_metrics_payload(str(agent_id), group))
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def _archetype_metrics(bids: pd.DataFrame) -> pd.DataFrame:
    if bids.empty or "archetype" not in bids.columns:
        return pd.DataFrame()
    rows = []
    for archetype, group in bids.groupby("archetype"):
        rows.append(_metrics_payload(str(archetype), group))
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def _run_summary(bids: pd.DataFrame, traces: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    run_id = str(bids["run_id"].iloc[0]) if not bids.empty else "unknown"
    payload = _metrics_payload(run_id, bids)
    payload["run_id"] = run_id
    payload.update(_watch_hour_metrics(traces, truth, actual_profit=float(bids["realized_profit_eur"].sum()) if not bids.empty else 0.0))
    payload.update(_asset_backend_comparison_metrics(traces))
    payload.update(_tool_autonomy_metrics(traces))
    payload["quantity_bucket_metrics"] = _quantity_bucket_metrics(bids)
    payload["accepted_candidate_rate_by_quantity_bucket"] = _accepted_candidate_rate_by_quantity_bucket(traces)
    return payload


def _metrics_payload(identifier: str, bids: pd.DataFrame) -> dict[str, Any]:
    filled = bids[bids["cleared_mwh"] > 0]
    bid_actions = bids[~bids["status"].isin(["abstain", "invalid", "gate_closed"])]
    verifier_accepted = bids[bids["verifier_accepted"] == True]  # noqa: E712
    profits = bids["realized_profit_eur"].astype(float)
    downside = profits[profits < 0]
    cumulative = profits.cumsum()
    running_max = cumulative.cummax() if not cumulative.empty else pd.Series(dtype=float)
    drawdown = cumulative - running_max if not cumulative.empty else pd.Series(dtype=float)
    accepted_breaches = verifier_accepted[verifier_accepted["realized_profit_eur"] < 0]
    interval_rows = bids[bids["forecast_interval_covered"].notna()]

    oracle_profit = _oracle_profit(bids)
    opportunity_volume = _opportunity_volume(bids)
    actual_profit = float(profits.sum()) if not bids.empty else 0.0
    return {
        "id": identifier,
        "bid_count": len(bids),
        "eligible_count": len(bid_actions),
        "filled_count": int((bids["status"] == "filled").sum()),
        "partially_filled_count": int((bids["status"] == "partially_filled").sum()),
        "cleared_mwh": round(float(bids["cleared_mwh"].sum()), 6) if not bids.empty else 0.0,
        "fill_rate": round(float(filled["cleared_mwh"].sum() / bid_actions["quantity_mwh"].sum()) if not bid_actions.empty and bid_actions["quantity_mwh"].sum() > 0 else 0.0, 6),
        "realized_profit_eur": round(actual_profit, 6),
        "profit_per_mwh": round(actual_profit / float(filled["cleared_mwh"].sum()), 6) if not filled.empty and filled["cleared_mwh"].sum() > 0 else None,
        "cumulative_pnl_eur": round(actual_profit, 6),
        "max_drawdown_eur": round(abs(float(drawdown.min())), 6) if not drawdown.empty else 0.0,
        "downside_cvar_95_eur": round(float(downside.nsmallest(max(1, int(len(downside) * 0.05))).mean()), 6) if not downside.empty else 0.0,
        "activation_direction_hit_rate": round(float((bid_actions["status"].isin(["filled", "partially_filled", "price_not_crossed", "no_activation"])).mean()), 6) if not bid_actions.empty else None,
        "side_precision": round(float((bid_actions["status"].isin(["filled", "partially_filled", "price_not_crossed", "no_activation"])).mean()), 6) if not bid_actions.empty else None,
        "volume_utilisation": round(float(filled["cleared_mwh"].sum() / opportunity_volume), 6) if opportunity_volume > 0 else 0.0,
        "mean_limit_price_distance_eur_mwh": round(float(bid_actions["price_distance_eur_mwh"].dropna().mean()), 6) if not bid_actions["price_distance_eur_mwh"].dropna().empty else None,
        "oracle_feasible_profit_eur": round(oracle_profit, 6),
        "oracle_feasible_profit_scope": "submitted_bid_conditional",
        "missed_profitable_activation_mwh": round(max(0.0, opportunity_volume - float(filled["cleared_mwh"].sum())), 6),
        "regret_eur": round(max(0.0, oracle_profit - actual_profit), 6),
        "opportunity_capture": round(actual_profit / oracle_profit, 6) if oracle_profit > 0 else None,
        "verifier_realized_profit_breach_rate": round(float(len(accepted_breaches) / len(verifier_accepted)), 6) if not verifier_accepted.empty else None,
        "verifier_false_accepts": len(accepted_breaches),
        "verifier_false_rejects": int(((bids["verifier_accepted"] == False) & (bids["status"].isin(["filled", "partially_filled"])) & (bids["realized_profit_eur"] > 0)).sum()),  # noqa: E712
        "realized_price_coverage": round(float(interval_rows["forecast_interval_covered"].astype(bool).mean()), 6) if not interval_rows.empty else None,
        "status_counts": {str(key): int(value) for key, value in bids["status"].value_counts().sort_index().items()},
        "watch_count": int((bids["status"] == "watch").sum()) if not bids.empty else 0,
        "bid_action_count": int((~bids["status"].isin(["abstain", "watch", "invalid", "gate_closed"])).sum()) if not bids.empty else 0,
        "watch_to_bid_conversion_rate": _watch_to_bid_conversion_rate(bids),
        "wrong_side_count": int((bids["status"] == "wrong_side").sum()) if not bids.empty else 0,
    }


def _watch_to_bid_conversion_rate(bids: pd.DataFrame) -> float:
    if bids.empty:
        return 0.0
    bid_count = int((~bids["status"].isin(["abstain", "watch", "invalid", "gate_closed"])).sum())
    watch_or_bid_count = int((bids["status"].isin(["watch"]) | ~bids["status"].isin(["abstain", "watch", "invalid", "gate_closed"])).sum())
    return round(float(bid_count / watch_or_bid_count), 6) if watch_or_bid_count > 0 else 0.0


def _quantity_bucket_metrics(bids: pd.DataFrame) -> dict[str, Any]:
    if bids.empty or "quantity_mwh" not in bids.columns:
        return {}
    rows = bids[pd.to_numeric(bids["quantity_mwh"], errors="coerce").notna()].copy()
    if rows.empty:
        return {}
    rows["quantity_bucket"] = rows["quantity_mwh"].astype(float).map(_quantity_bucket)
    out: dict[str, Any] = {}
    for bucket, group in rows.groupby("quantity_bucket", sort=False):
        bid_actions = group[~group["status"].isin(["abstain", "watch", "invalid", "gate_closed"])]
        submitted_mwh = float(bid_actions["quantity_mwh"].sum()) if not bid_actions.empty else 0.0
        cleared_mwh = float(group["cleared_mwh"].sum())
        realized_profit = float(group["realized_profit_eur"].sum())
        out[str(bucket)] = {
            "bid_count": int(len(group)),
            "bid_action_count": int(len(bid_actions)),
            "submitted_mwh": round(submitted_mwh, 6),
            "cleared_mwh": round(cleared_mwh, 6),
            "fill_rate": round(cleared_mwh / submitted_mwh, 6) if submitted_mwh > 0 else 0.0,
            "realized_profit_eur": round(realized_profit, 6),
            "profit_per_mwh": round(realized_profit / cleared_mwh, 6) if cleared_mwh > 0 else None,
            "wrong_side_count": int((group["status"] == "wrong_side").sum()),
            "price_not_crossed_count": int((group["status"] == "price_not_crossed").sum()),
        }
    return out


def _accepted_candidate_rate_by_quantity_bucket(traces: pd.DataFrame) -> dict[str, Any]:
    if traces.empty or "tool_calls" not in traces.columns:
        return {}
    buckets: dict[str, dict[str, int]] = {}
    for _, trace in traces.iterrows():
        tool_calls = trace.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for record in tool_calls:
            if not isinstance(record, dict):
                continue
            name = str(record.get("name") or "")
            if not name.startswith("simulate"):
                continue
            arguments = record.get("arguments") if isinstance(record.get("arguments"), dict) else {}
            result = record.get("result") if isinstance(record.get("result"), dict) else {}
            try:
                quantity = float(arguments.get("quantity_mwh"))
            except (TypeError, ValueError):
                continue
            bucket = _quantity_bucket(quantity)
            row = buckets.setdefault(bucket, {"simulated_count": 0, "accepted_count": 0})
            row["simulated_count"] += 1
            if result.get("accepted") is True:
                row["accepted_count"] += 1
    return {
        bucket: {
            **counts,
            "accepted_rate": round(counts["accepted_count"] / counts["simulated_count"], 6) if counts["simulated_count"] else 0.0,
        }
        for bucket, counts in sorted(buckets.items())
    }


def _quantity_bucket(quantity_mwh: float) -> str:
    if quantity_mwh <= 0.5 + 1e-9:
        return "q_le_0_5"
    if quantity_mwh <= 2.0 + 1e-9:
        return "q_0_5_2"
    if quantity_mwh <= 4.0 + 1e-9:
        return "q_2_4"
    if quantity_mwh <= 10.0 + 1e-9:
        return "q_4_10"
    return "q_gt_10"


def _watch_hour_metrics(traces: pd.DataFrame, truth: pd.DataFrame, *, actual_profit: float) -> dict[str, Any]:
    if traces.empty:
        return {
            "truth_window_oracle_profit_eur": 0.0,
            "truth_window_oracle_capture": None,
            "missed_truth_window_oracle_profit_eur": 0.0,
            "activated_tick_count": 0,
            "profitable_tick_count": 0,
        }
    ticks = traces[["timestamp_utc", "zone"]].drop_duplicates()
    truth_window = ticks.merge(truth, on=["timestamp_utc", "zone"], how="left")
    truth_window["truth_profit_per_mwh"] = truth_window.apply(_truth_profit_per_mwh, axis=1)
    truth_window["truth_oracle_profit_eur"] = truth_window["truth_profit_per_mwh"].clip(lower=0) * truth_window["activated_volume_mwh"].fillna(0.0)
    truth_window["is_activated"] = truth_window["activation_direction"].isin(["up", "down"]) & (truth_window["activated_volume_mwh"].fillna(0.0) > 0)
    truth_window["is_profitable"] = truth_window["truth_oracle_profit_eur"] > 0

    actions = (
        traces.assign(
            is_watch=lambda frame: frame["action"] == "watch",
            is_bid=lambda frame: frame["action"] == "bid",
            is_watch_or_bid=lambda frame: frame["action"].isin(["watch", "bid"]),
            is_must_watch=lambda frame: frame["watch_label"] == "must_watch",
            is_label_watch_or_bid=lambda frame: frame["watch_label"].isin(["must_watch", "watch"]) | frame["action"].isin(["watch", "bid"]),
        )
        .groupby(["timestamp_utc", "zone"], sort=False)
        .agg(
            any_watch=("is_watch", "any"),
            any_bid=("is_bid", "any"),
            any_watch_or_bid=("is_watch_or_bid", "any"),
            any_must_watch=("is_must_watch", "any"),
            any_label_watch_or_bid=("is_label_watch_or_bid", "any"),
            watch_vote_share=("is_label_watch_or_bid", "mean"),
        )
        .reset_index()
    )
    joined = truth_window.merge(actions, on=["timestamp_utc", "zone"], how="left")
    for column in ["any_watch", "any_bid", "any_watch_or_bid", "any_must_watch", "any_label_watch_or_bid"]:
        joined[column] = joined[column].fillna(False).astype(bool)
    joined["watch_vote_share"] = joined["watch_vote_share"].fillna(0.0)

    activated = joined["is_activated"]
    profitable = joined["is_profitable"]
    watch = joined["any_watch"]
    bid = joined["any_bid"]
    watch_or_bid = joined["any_watch_or_bid"]
    must_watch = joined["any_must_watch"]
    label_watch_or_bid = joined["any_label_watch_or_bid"]
    quiet = ~(activated | profitable)
    oracle = float(joined["truth_oracle_profit_eur"].sum())
    reason_count = sum(len(reasons) for reasons in traces["watch_reasons"] if isinstance(reasons, list))
    valid_reason_count = sum(
        1
        for reasons in traces["watch_reasons"]
        if isinstance(reasons, list)
        for reason in reasons
        if reason in {
            "activation_risk",
            "price_volatility",
            "forecast_uncertainty",
            "accepted_bid_available",
            "verifier_rejection_cluster",
            "cross_agent_disagreement",
        }
    )
    confidence_dispersion = float(traces["confidence"].dropna().std(ddof=0)) if "confidence" in traces and not traces["confidence"].dropna().empty else None
    return {
        "truth_window_oracle_profit_eur": round(oracle, 6),
        "truth_window_oracle_capture": round(actual_profit / oracle, 6) if oracle > 0 else None,
        "missed_truth_window_oracle_profit_eur": round(max(0.0, oracle - actual_profit), 6),
        "activated_tick_count": int(activated.sum()),
        "profitable_tick_count": int(profitable.sum()),
        "watch_tick_count": int(watch.sum()),
        "bid_tick_count": int(bid.sum()),
        "watch_or_bid_tick_count": int(watch_or_bid.sum()),
        "must_watch_tick_count": int(must_watch.sum()),
        "label_watch_or_bid_tick_count": int(label_watch_or_bid.sum()),
        "activated_watch_or_bid_recall": _safe_rate((watch_or_bid & activated).sum(), activated.sum()),
        "activated_watch_or_bid_precision": _safe_rate((watch_or_bid & activated).sum(), watch_or_bid.sum()),
        "profitable_watch_or_bid_recall": _safe_rate((watch_or_bid & profitable).sum(), profitable.sum()),
        "profitable_watch_or_bid_precision": _safe_rate((watch_or_bid & profitable).sum(), watch_or_bid.sum()),
        "profitable_watch_recall": _safe_rate((watch & profitable).sum(), profitable.sum()),
        "profitable_watch_precision": _safe_rate((watch & profitable).sum(), watch.sum()),
        "profitable_bid_recall": _safe_rate((bid & profitable).sum(), profitable.sum()),
        "profitable_bid_precision": _safe_rate((bid & profitable).sum(), bid.sum()),
        "must_watch_profitable_or_activated_recall": _safe_rate((must_watch & (profitable | activated)).sum(), (profitable | activated).sum()),
        "must_watch_profitable_or_activated_precision": _safe_rate((must_watch & (profitable | activated)).sum(), must_watch.sum()),
        "label_watch_or_bid_recall": _safe_rate((label_watch_or_bid & (profitable | activated)).sum(), (profitable | activated).sum()),
        "alert_spam_rate": _safe_rate((label_watch_or_bid & quiet).sum(), label_watch_or_bid.sum()),
        "quiet_hour_abstain_correctness": _safe_rate((~label_watch_or_bid & quiet).sum(), quiet.sum()),
        "consensus_watch_quality": round(float(joined.loc[profitable | activated, "watch_vote_share"].mean()), 6) if (profitable | activated).any() else None,
        "contested_watch_count": int(((joined["watch_vote_share"] > 0.0) & (joined["watch_vote_share"] < 1.0)).sum()),
        "reason_diversity": int(len({reason for reasons in traces["watch_reasons"] if isinstance(reasons, list) for reason in reasons})),
        "confidence_dispersion": round(confidence_dispersion, 6) if confidence_dispersion is not None else None,
        "evidence_reference_validity": _safe_rate(valid_reason_count, reason_count),
    }


def _truth_profit_per_mwh(row: pd.Series) -> float:
    if row.get("activation_direction") == "up":
        return float(row["settlement_price_eur_mwh"]) - float(row["spot_price_eur_mwh"])
    if row.get("activation_direction") == "down":
        return float(row["spot_price_eur_mwh"]) - float(row["settlement_price_eur_mwh"])
    return 0.0


def _safe_rate(numerator: Any, denominator: Any) -> float | None:
    denominator = int(denominator)
    if denominator <= 0:
        return None
    return round(float(numerator) / denominator, 6)


def _oracle_profit(bids: pd.DataFrame) -> float:
    if bids.empty:
        return 0.0
    opportunities = bids[["timestamp_utc", "zone", "side", "activated_volume_mwh", "profit_per_mwh"]].dropna().drop_duplicates()
    positive = opportunities[opportunities["profit_per_mwh"] > 0]
    return round(float((positive["activated_volume_mwh"] * positive["profit_per_mwh"]).sum()), 6)


def _opportunity_volume(bids: pd.DataFrame) -> float:
    if bids.empty:
        return 0.0
    opportunities = bids[["timestamp_utc", "zone", "side", "activated_volume_mwh", "profit_per_mwh"]].dropna().drop_duplicates()
    positive = opportunities[opportunities["profit_per_mwh"] > 0]
    return round(float(positive["activated_volume_mwh"].sum()), 6)


def _load_traces(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        decision = payload.get("decision") or {}
        forecast_interval = payload.get("forecast_interval_eur_mwh") or [None, None]
        rows.append(
            {
                "run_id": payload.get("run_id"),
                "step": payload.get("step"),
                "timestamp_utc": payload.get("timestamp"),
                "agent_id": payload.get("agent_id"),
                "archetype": payload.get("archetype"),
                "agent_role": payload.get("agent_role", "action_agent"),
                "zone": payload.get("zone") or decision.get("zone") or "DK1",
                "action": decision.get("action"),
                "watch_label": decision.get("watch_label", "watch" if decision.get("action") == "watch" else "ignore"),
                "risk_label": decision.get("risk_label", "low"),
                "uncertainty_label": decision.get("uncertainty_label", "low"),
                "opportunity_label": decision.get("opportunity_label", "none"),
                "watch_reasons": decision.get("watch_reasons") or [],
                "priority_label": decision.get("priority_label", "low"),
                "priority_score": decision.get("priority_score", 0.0),
                "operator_action": decision.get("operator_action", "ignore"),
                "priority_reason": decision.get("priority_reason", "none"),
                "confidence": decision.get("confidence"),
                "side": decision.get("side"),
                "quantity_mwh": decision.get("quantity_mwh"),
                "limit_price_eur_mwh": decision.get("limit_price_eur_mwh"),
                "submitted_at_utc": decision.get("submitted_at_utc"),
                "verifier_accepted": payload.get("verifier_accepted"),
                "forecast_lower_eur_mwh": forecast_interval[0],
                "forecast_upper_eur_mwh": forecast_interval[1],
                "tool_calls": payload.get("tool_calls") or [],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp_utc", "zone"])
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["submitted_at_utc"] = pd.to_datetime(frame["submitted_at_utc"], utc=True, errors="coerce")
    for column in ["quantity_mwh", "limit_price_eur_mwh", "forecast_lower_eur_mwh", "forecast_upper_eur_mwh"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _asset_backend_comparison_metrics(traces: pd.DataFrame) -> dict[str, Any]:
    if traces.empty or "tool_calls" not in traces.columns:
        return {
            "asset_backend_comparison_count": 0,
            "asset_backend_disagreement_rate": None,
            "asset_backend_proxy_false_positive_rate": None,
            "asset_backend_scenario_envelope_false_positive_rate": None,
            "asset_backend_comparison_by_archetype": {},
        }
    rows: list[dict[str, Any]] = []
    for _, trace in traces.iterrows():
        tool_calls = trace.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for record in tool_calls:
            if not isinstance(record, dict):
                continue
            name = str(record.get("name") or "")
            result = record.get("result") or {}
            comparison = result.get("comparison") if isinstance(result, dict) else None
            if not name.startswith("simulate") or not isinstance(comparison, dict):
                continue
            proxy = _comparison_backend(comparison, "proxy")
            scenario_envelope = _comparison_backend(comparison, "scenario_envelope", "scenario_asset_v1")
            pypsa_background = _comparison_backend(comparison, "pypsa_background", "p2h_pypsa_mfrr")
            rows.append(
                {
                    "archetype": str(trace.get("archetype") or result.get("archetype") or ""),
                    "proxy_accepted": proxy.get("accepted") is True,
                    "scenario_envelope_accepted": scenario_envelope.get("accepted") is True,
                    "pypsa_background_accepted": pypsa_background.get("accepted") is True,
                    "accepted_disagreement": comparison.get("accepted_disagreement") is True,
                    "proxy_false_positive": comparison.get("proxy_false_positive") is True,
                    "scenario_envelope_false_positive": comparison.get("scenario_envelope_false_positive") is True,
                    "proxy_worst_case_profit_eur": _safe_float(proxy.get("worst_case_profit_eur")),
                    "scenario_envelope_worst_case_profit_eur": _safe_float(scenario_envelope.get("worst_case_profit_eur")),
                    "pypsa_background_worst_case_profit_eur": _safe_float(pypsa_background.get("worst_case_profit_eur")),
                }
            )
    if not rows:
        return {
            "asset_backend_comparison_count": 0,
            "asset_backend_disagreement_rate": None,
            "asset_backend_proxy_false_positive_rate": None,
            "asset_backend_scenario_envelope_false_positive_rate": None,
            "asset_backend_comparison_by_archetype": {},
        }
    frame = pd.DataFrame(rows)
    by_archetype = {}
    for archetype, group in frame.groupby("archetype"):
        by_archetype[str(archetype)] = _asset_backend_payload(group)
    return {
        **_asset_backend_payload(frame, prefix="asset_backend_"),
        "asset_backend_comparison_by_archetype": by_archetype,
    }


def _tool_autonomy_metrics(traces: pd.DataFrame) -> dict[str, Any]:
    if traces.empty or "tool_calls" not in traces.columns:
        return _empty_tool_autonomy_metrics()
    total_calls = 0
    llm_calls = 0
    forced_calls = 0
    simulator_calls = 0
    llm_simulator_calls = 0
    unsupported_bid_proposals = 0
    accepted_bids = 0
    accepted_bids_backed_by_llm_simulator = 0
    trace_count = len(traces)
    provenance_counts = {
        "runner_seeded": 0,
        "llm_requested": 0,
        "forced_final": 0,
        "runner_diagnostic": 0,
        "retry": 0,
        "unknown": 0,
    }

    for _, trace in traces.iterrows():
        calls = trace.get("tool_calls")
        if not isinstance(calls, list):
            calls = []
        total_calls += len(calls)
        for call in calls:
            if not isinstance(call, dict):
                continue
            provenance = str(call.get("provenance") or "unknown")
            if provenance not in provenance_counts:
                provenance = "unknown"
            provenance_counts[provenance] += 1
            name = str(call.get("name") or "")
            if provenance == "llm_requested":
                llm_calls += 1
            if provenance == "forced_final":
                forced_calls += 1
            if name.startswith("simulate"):
                simulator_calls += 1
                if provenance == "llm_requested":
                    llm_simulator_calls += 1
        decision = trace.get("decision") if isinstance(trace.get("decision"), dict) else {
            "action": trace.get("action"),
            "side": trace.get("side"),
            "quantity_mwh": trace.get("quantity_mwh"),
            "limit_price_eur_mwh": trace.get("limit_price_eur_mwh"),
        }
        if decision.get("action") == "bid":
            match = _matching_trace_simulator_call(decision, calls)
            if match is None:
                unsupported_bid_proposals += 1
            if trace.get("verifier_accepted") == True:  # noqa: E712
                accepted_bids += 1
                if match is not None and match.get("provenance") == "llm_requested":
                    accepted_bids_backed_by_llm_simulator += 1

    return {
        "tool_call_provenance_counts": provenance_counts,
        "autonomous_tool_call_rate": round(llm_calls / total_calls, 6) if total_calls else 0.0,
        "simulator_self_call_rate": round(llm_simulator_calls / simulator_calls, 6) if simulator_calls else 0.0,
        "unsupported_bid_proposal_rate": round(unsupported_bid_proposals / trace_count, 6) if trace_count else 0.0,
        "accepted_bid_backed_by_llm_requested_simulator_rate": (
            round(accepted_bids_backed_by_llm_simulator / accepted_bids, 6) if accepted_bids else None
        ),
        "extra_tool_rounds_per_decision": round(llm_calls / trace_count, 6) if trace_count else 0.0,
        "final_action_forced_rate": round(forced_calls / trace_count, 6) if trace_count else 0.0,
        "llm_tool_call_count": llm_calls,
        "forced_tool_call_count": forced_calls,
        "llm_requested_simulator_call_count": llm_simulator_calls,
        "unsupported_bid_proposal_count": unsupported_bid_proposals,
        "accepted_bid_backed_by_llm_requested_simulator_count": accepted_bids_backed_by_llm_simulator,
    }


def _empty_tool_autonomy_metrics() -> dict[str, Any]:
    return {
        "tool_call_provenance_counts": {},
        "autonomous_tool_call_rate": 0.0,
        "simulator_self_call_rate": 0.0,
        "unsupported_bid_proposal_rate": 0.0,
        "accepted_bid_backed_by_llm_requested_simulator_rate": None,
        "extra_tool_rounds_per_decision": 0.0,
        "final_action_forced_rate": 0.0,
        "llm_tool_call_count": 0,
        "forced_tool_call_count": 0,
        "llm_requested_simulator_call_count": 0,
        "unsupported_bid_proposal_count": 0,
        "accepted_bid_backed_by_llm_requested_simulator_count": 0,
    }


def _matching_trace_simulator_call(decision: dict[str, Any], calls: list[Any]) -> dict[str, Any] | None:
    side = decision.get("side")
    quantity = _safe_float(decision.get("quantity_mwh"))
    limit = _safe_float(decision.get("limit_price_eur_mwh"))
    if side not in {"up", "down"} or quantity is None or limit is None:
        return None
    for call in calls:
        if not isinstance(call, dict):
            continue
        if not str(call.get("name") or "").startswith("simulate"):
            continue
        result = call.get("result") if isinstance(call.get("result"), dict) else {}
        if result.get("accepted") is not True:
            continue
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if arguments.get("side") != side:
            continue
        if abs((_safe_float(arguments.get("quantity_mwh")) or -1.0) - quantity) > 1e-9:
            continue
        if abs((_safe_float(arguments.get("limit_price_eur_mwh")) or -1e9) - limit) > 1e-9:
            continue
        return call
    return None


def _asset_backend_payload(frame: pd.DataFrame, *, prefix: str = "") -> dict[str, Any]:
    return {
        f"{prefix}comparison_count": int(len(frame)),
        f"{prefix}proxy_accepted_count": int(frame["proxy_accepted"].sum()),
        f"{prefix}scenario_envelope_accepted_count": int(frame["scenario_envelope_accepted"].sum()),
        f"{prefix}pypsa_background_accepted_count": int(frame["pypsa_background_accepted"].sum()),
        f"{prefix}real_accepted_count": int(frame["scenario_envelope_accepted"].sum()),
        f"{prefix}accepted_disagreement_count": int(frame["accepted_disagreement"].sum()),
        f"{prefix}disagreement_rate": round(float(frame["accepted_disagreement"].mean()), 6),
        f"{prefix}proxy_false_positive_count": int(frame["proxy_false_positive"].sum()),
        f"{prefix}proxy_false_positive_rate": round(float(frame["proxy_false_positive"].mean()), 6),
        f"{prefix}scenario_envelope_false_positive_count": int(frame["scenario_envelope_false_positive"].sum()),
        f"{prefix}scenario_envelope_false_positive_rate": round(float(frame["scenario_envelope_false_positive"].mean()), 6),
        f"{prefix}mean_proxy_worst_case_profit_eur": round(float(frame["proxy_worst_case_profit_eur"].dropna().mean()), 6) if not frame["proxy_worst_case_profit_eur"].dropna().empty else None,
        f"{prefix}mean_scenario_envelope_worst_case_profit_eur": round(float(frame["scenario_envelope_worst_case_profit_eur"].dropna().mean()), 6) if not frame["scenario_envelope_worst_case_profit_eur"].dropna().empty else None,
        f"{prefix}mean_pypsa_background_worst_case_profit_eur": round(float(frame["pypsa_background_worst_case_profit_eur"].dropna().mean()), 6) if not frame["pypsa_background_worst_case_profit_eur"].dropna().empty else None,
        f"{prefix}mean_real_worst_case_profit_eur": round(float(frame["scenario_envelope_worst_case_profit_eur"].dropna().mean()), 6) if not frame["scenario_envelope_worst_case_profit_eur"].dropna().empty else None,
    }


def _comparison_backend(comparison: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = comparison.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_truth(path: Path) -> pd.DataFrame:
    truth = pd.read_parquet(path)
    required = {
        "timestamp_utc",
        "zone",
        "activation_direction",
        "activated_volume_mwh",
        "spot_price_eur_mwh",
        "settlement_price_eur_mwh",
    }
    missing = sorted(required - set(truth.columns))
    if missing:
        raise ValueError(f"evaluation truth missing required columns: {missing}")
    truth = truth.copy()
    truth["timestamp_utc"] = pd.to_datetime(truth["timestamp_utc"], utc=True)
    return truth


def _read_manifest(path: Path, *, expected_visibility: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    visibility = manifest.get("visibility")
    if visibility != expected_visibility:
        raise RuntimeError(f"{path} has visibility={visibility!r}, expected {expected_visibility!r}")
    return manifest


def _interval_covered(trace: pd.Series, settlement_price: float) -> bool | None:
    lower = trace.get("forecast_lower_eur_mwh")
    upper = trace.get("forecast_upper_eur_mwh")
    if pd.isna(lower) or pd.isna(upper):
        return None
    lo, hi = sorted((float(lower), float(upper)))
    return lo <= settlement_price <= hi


def _settlement(*, side: str, quantity_mwh: float, spot_price: float, clearing_price: float) -> float:
    return quantity_mwh * _profit_per_mwh(side=side, spot_price=spot_price, clearing_price=clearing_price)


def _profit_per_mwh(*, side: str, spot_price: float, clearing_price: float) -> float:
    if side == "up":
        return clearing_price - spot_price
    return spot_price - clearing_price


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
