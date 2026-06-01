from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_RUN_IDS = [
    "bme-promote-jao12-bcast-mem-apr02-0530-48-f8",
    "bme-jao12-bcast-mem-apr09-1830-24-f8",
    "bme-info14-bcast-mem-apr02-0530-24-f8",
    "bme-info14-bcast-mem-apr02-0530-24-f3_ensemble",
]

STATUS_COLUMNS = ["filled", "wrong_side", "price_not_crossed", "no_activation", "watch", "abstain"]
PRICE_COLUMNS = [
    "settlement_price_eur_mwh",
    "spot_price_eur_mwh",
    "imbalance_price_eur_mwh",
    "mfrr_up_price_eur_mwh",
    "mfrr_down_price_eur_mwh",
]


@dataclass(frozen=True)
class RunBundle:
    run_id: str
    output_dir: Path
    summary: dict[str, Any]
    ticks: pd.DataFrame
    truth: pd.DataFrame
    jao: pd.DataFrame | None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot recent Heimdall run quality against activation truth."
    )
    parser.add_argument("--run-id", action="append", dest="run_ids", help="Run id to plot.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to reports/recent-run-plots/<timestamp>.")
    parser.add_argument("--evaluations-dir", type=Path, default=Path("evaluations"))
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("ai-society/runs"),
        help="Root searched recursively for <run_id>/traces.jsonl.",
    )
    parser.add_argument(
        "--truth-path",
        type=Path,
        default=Path("data/cache/evaluation_truth/april_2026/activation_truth.parquet"),
    )
    parser.add_argument(
        "--jao-path",
        type=Path,
        default=Path("data/cache/real_context/april_2026/jao_constraints.parquet"),
    )
    args = parser.parse_args()

    run_ids = args.run_ids or DEFAULT_RUN_IDS
    output_dir = args.output_dir or Path("reports/recent-run-plots") / _timestamp_slug()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = [
        _load_run_bundle(
            run_id=run_id,
            output_root=output_dir,
            evaluations_dir=args.evaluations_dir,
            runs_root=args.runs_root,
            truth_path=args.truth_path,
            jao_path=args.jao_path,
        )
        for run_id in run_ids
    ]

    for bundle in bundles:
        _plot_market_truth(bundle)
        _plot_system_actions(bundle)
        _plot_overlay_quality(bundle)
        _plot_profit_and_errors(bundle)
        _plot_jao_context(bundle)

    _plot_summary_comparison(bundles, output_dir / "summary_comparison.png")
    print(json.dumps({"ok": True, "output_dir": str(output_dir), "runs": run_ids}, indent=2))


def _load_run_bundle(
    *,
    run_id: str,
    output_root: Path,
    evaluations_dir: Path,
    runs_root: Path,
    truth_path: Path,
    jao_path: Path,
) -> RunBundle:
    evaluation_dir = evaluations_dir / run_id
    bid_path = evaluation_dir / "bid_evaluations.parquet"
    summary_path = evaluation_dir / "run_summary.json"
    trace_path = _find_trace_path(runs_root, run_id)
    _require_file(bid_path)
    _require_file(summary_path)
    _require_file(trace_path)
    _require_file(truth_path)

    bids = pd.read_parquet(bid_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    traces = _load_traces(trace_path)
    ticks = _aggregate_ticks(bids, traces)

    if ticks.empty:
        raise RuntimeError(f"no tick data found for {run_id}")

    truth = pd.read_parquet(truth_path)
    truth = truth[truth["zone"].eq("DK1")].copy()
    truth["timestamp_utc"] = pd.to_datetime(truth["timestamp_utc"], utc=True)
    truth = ticks[["timestamp_utc"]].merge(truth, on="timestamp_utc", how="left")
    if truth[[*PRICE_COLUMNS, "activated_volume_mwh"]].isna().all(axis=None):
        raise RuntimeError(f"no activation truth joined for {run_id}")

    jao = _load_jao(jao_path, ticks["timestamp_utc"]) if jao_path.exists() else None
    run_output_dir = output_root / _safe_name(run_id)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    return RunBundle(
        run_id=run_id,
        output_dir=run_output_dir,
        summary=summary,
        ticks=ticks,
        truth=truth,
        jao=jao,
    )


def _find_trace_path(runs_root: Path, run_id: str) -> Path:
    direct = runs_root / run_id / "traces.jsonl"
    if direct.exists():
        return direct
    matches = sorted(runs_root.glob(f"**/{run_id}/traces.jsonl"))
    if not matches:
        raise FileNotFoundError(f"trace file missing for {run_id} under {runs_root}")
    if len(matches) > 1:
        print(f"warning: multiple traces for {run_id}; using {matches[0]}")
    return matches[0]


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required artifact missing: {path}")


def _load_traces(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            decision = _parse_decision(row.get("decision"))
            row["decision_action"] = decision.get("action")
            row["decision_watch_label"] = decision.get("watch_label")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"trace file is empty: {path}")
    traces = pd.DataFrame(rows)
    traces["timestamp_utc"] = pd.to_datetime(traces["timestamp"], utc=True)
    return traces


def _parse_decision(decision: Any) -> dict[str, Any]:
    if isinstance(decision, dict):
        return decision
    if isinstance(decision, str):
        try:
            parsed = ast.literal_eval(decision)
        except (SyntaxError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _aggregate_ticks(bids: pd.DataFrame, traces: pd.DataFrame) -> pd.DataFrame:
    bids = bids.copy()
    bids["timestamp_utc"] = pd.to_datetime(bids["timestamp_utc"], utc=True)
    traces = traces.copy()
    traces["timestamp_utc"] = pd.to_datetime(traces["timestamp_utc"], utc=True)

    status_counts = (
        bids.pivot_table(
            index="timestamp_utc",
            columns="status",
            values="agent_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=STATUS_COLUMNS, fill_value=0)
        .reset_index()
    )
    bid_sums = (
        bids.groupby("timestamp_utc", as_index=False)
        .agg(
            realized_profit_eur=("realized_profit_eur", "sum"),
            cleared_mwh=("cleared_mwh", "sum"),
            evaluated_activated_volume_mwh=("activated_volume_mwh", "max"),
        )
    )
    trace_counts = (
        traces.groupby("timestamp_utc", as_index=False)
        .agg(
            agent_count=("agent_id", "count"),
            watch_action_count=("decision_action", lambda s: int(s.eq("watch").sum())),
            bid_action_count=("decision_action", lambda s: int(s.eq("bid").sum())),
            abstain_action_count=("decision_action", lambda s: int(s.eq("abstain").sum())),
            must_watch_count=("decision_watch_label", lambda s: int(s.eq("must_watch").sum())),
        )
    )

    ticks = status_counts.merge(bid_sums, on="timestamp_utc", how="left")
    ticks = ticks.merge(trace_counts, on="timestamp_utc", how="left")
    ticks = ticks.sort_values("timestamp_utc").reset_index(drop=True)
    for column in [
        *STATUS_COLUMNS,
        "realized_profit_eur",
        "cleared_mwh",
        "evaluated_activated_volume_mwh",
        "agent_count",
        "watch_action_count",
        "bid_action_count",
        "abstain_action_count",
        "must_watch_count",
    ]:
        ticks[column] = ticks[column].fillna(0)

    ticks["must_watch_share"] = ticks["must_watch_count"] / ticks["agent_count"].clip(lower=1)
    ticks["high_must_watch"] = ticks["must_watch_share"].ge(0.5)
    bid_threshold = ticks["bid_action_count"].quantile(0.75)
    ticks["bid_heavy"] = ticks["bid_action_count"].ge(bid_threshold) & ticks["bid_action_count"].gt(0)
    ticks["cumulative_profit_eur"] = ticks["realized_profit_eur"].cumsum()
    return ticks


def _load_jao(path: Path, timestamps: pd.Series) -> pd.DataFrame:
    start = timestamps.min()
    end = timestamps.max()
    columns = [
        "timestamp_utc",
        "zone",
        "ram_mw",
        "shadow_price_eur_mw",
        "flow_mw",
    ]
    jao = pd.read_parquet(path, columns=columns)
    jao["timestamp_utc"] = pd.to_datetime(jao["timestamp_utc"], utc=True)
    jao = jao[jao["zone"].eq("DK1") & jao["timestamp_utc"].between(start, end)].copy()
    if jao.empty:
        return pd.DataFrame(
            columns=["timestamp_utc", "max_shadow_price_eur_mw", "min_ram_mw", "max_abs_flow_mw"]
        )
    jao["abs_flow_mw"] = jao["flow_mw"].abs()
    return (
        jao.groupby("timestamp_utc", as_index=False)
        .agg(
            max_shadow_price_eur_mw=("shadow_price_eur_mw", "max"),
            min_ram_mw=("ram_mw", "min"),
            max_abs_flow_mw=("abs_flow_mw", "max"),
        )
        .sort_values("timestamp_utc")
    )


def _plot_market_truth(bundle: RunBundle) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    truth = bundle.truth
    for column in PRICE_COLUMNS:
        axes[0].plot(truth["timestamp_utc"], truth[column], label=_label(column), linewidth=1.8)
    axes[0].set_title(f"{bundle.run_id}: market prices and activation truth")
    axes[0].set_ylabel("EUR/MWh")
    axes[0].legend(loc="upper left", ncols=3, fontsize=8)
    axes[0].grid(alpha=0.25)

    colors = {"up": "#d62728", "down": "#1f77b4", "neutral": "#7f7f7f"}
    axes[1].bar(
        truth["timestamp_utc"],
        truth["activated_volume_mwh"],
        color=[colors.get(str(v), "#7f7f7f") for v in truth["activation_direction"]],
        width=0.008,
    )
    axes[1].set_ylabel("Activated MWh")
    axes[1].grid(alpha=0.25)

    direction_map = {"down": -1, "neutral": 0, "up": 1}
    axes[2].step(
        truth["timestamp_utc"],
        truth["activation_direction"].map(direction_map).fillna(0),
        where="mid",
        color="#333333",
    )
    axes[2].set_yticks([-1, 0, 1], ["down", "neutral", "up"])
    axes[2].set_ylabel("Direction")
    axes[2].grid(alpha=0.25)
    _format_time_axis(axes[2])
    _save(fig, bundle.output_dir / "01_market_truth.png")


def _plot_system_actions(bundle: RunBundle) -> None:
    ticks = bundle.ticks
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].bar(ticks["timestamp_utc"], ticks["bid_action_count"], label="bid actions", width=0.008)
    axes[0].bar(
        ticks["timestamp_utc"],
        ticks["watch_action_count"],
        bottom=ticks["bid_action_count"],
        label="watch actions",
        width=0.008,
    )
    axes[0].bar(
        ticks["timestamp_utc"],
        ticks["abstain_action_count"],
        bottom=ticks["bid_action_count"] + ticks["watch_action_count"],
        label="abstain actions",
        width=0.008,
    )
    axes[0].set_title(f"{bundle.run_id}: system actions per tick")
    axes[0].set_ylabel("Agent decisions")
    axes[0].legend(loc="upper left", ncols=3, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(ticks["timestamp_utc"], ticks["must_watch_count"], marker="o", label="must watch")
    axes[1].plot(ticks["timestamp_utc"], ticks["watch_action_count"], marker="o", label="watch action")
    axes[1].set_ylabel("Count")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(alpha=0.25)

    for status in ["filled", "wrong_side", "price_not_crossed", "no_activation"]:
        axes[2].plot(ticks["timestamp_utc"], ticks[status], marker="o", label=status)
    axes[2].set_ylabel("Evaluation count")
    axes[2].legend(loc="upper left", ncols=4, fontsize=8)
    axes[2].grid(alpha=0.25)
    _format_time_axis(axes[2])
    _save(fig, bundle.output_dir / "02_system_actions.png")


def _plot_overlay_quality(bundle: RunBundle) -> None:
    ticks = bundle.ticks
    truth = bundle.truth
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    _shade_high_must_watch(axes[0], ticks)
    axes[0].plot(
        truth["timestamp_utc"],
        truth["settlement_price_eur_mwh"],
        color="#111111",
        label="settlement",
    )
    axes[0].plot(truth["timestamp_utc"], truth["spot_price_eur_mwh"], color="#2ca02c", label="spot")
    bid_heavy = ticks[ticks["bid_heavy"]]
    axes[0].scatter(
        bid_heavy["timestamp_utc"],
        truth.set_index("timestamp_utc").loc[bid_heavy["timestamp_utc"], "settlement_price_eur_mwh"],
        color="#ff7f0e",
        marker="^",
        s=70,
        label="bid-heavy tick",
        zorder=5,
    )
    axes[0].set_title(f"{bundle.run_id}: truth with must-watch and bid-heavy overlays")
    axes[0].set_ylabel("EUR/MWh")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(alpha=0.25)

    _shade_high_must_watch(axes[1], ticks)
    axes[1].bar(
        truth["timestamp_utc"],
        truth["activated_volume_mwh"],
        width=0.008,
        color="#7f7f7f",
        label="activated volume",
    )
    axes[1].plot(
        ticks["timestamp_utc"],
        ticks["must_watch_count"],
        color="#d62728",
        marker="o",
        label="must-watch count",
    )
    axes[1].plot(
        ticks["timestamp_utc"],
        ticks["bid_action_count"],
        color="#ff7f0e",
        marker="o",
        label="bid count",
    )
    axes[1].set_ylabel("MWh / count")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(alpha=0.25)
    _format_time_axis(axes[1])
    _save(fig, bundle.output_dir / "03_overlay_quality.png")


def _plot_profit_and_errors(bundle: RunBundle) -> None:
    ticks = bundle.ticks
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in ticks["realized_profit_eur"]]
    axes[0].bar(ticks["timestamp_utc"], ticks["realized_profit_eur"], color=colors, width=0.008)
    axes[0].plot(
        ticks["timestamp_utc"],
        ticks["cumulative_profit_eur"],
        color="#111111",
        marker="o",
        label="cumulative profit",
    )
    axes[0].set_title(f"{bundle.run_id}: profit and error modes")
    axes[0].set_ylabel("EUR")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(alpha=0.25)

    bottom = pd.Series([0] * len(ticks), index=ticks.index)
    for status, color in [
        ("wrong_side", "#d62728"),
        ("price_not_crossed", "#9467bd"),
        ("no_activation", "#8c564b"),
        ("filled", "#2ca02c"),
    ]:
        axes[1].bar(
            ticks["timestamp_utc"],
            ticks[status],
            bottom=bottom,
            width=0.008,
            label=status,
            color=color,
        )
        bottom = bottom + ticks[status]
    axes[1].set_ylabel("Evaluation count")
    axes[1].legend(loc="upper left", ncols=4, fontsize=8)
    axes[1].grid(alpha=0.25)
    _format_time_axis(axes[1])
    _save(fig, bundle.output_dir / "04_profit_and_errors.png")


def _plot_jao_context(bundle: RunBundle) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(
        f"{bundle.run_id}: JAO context for inspection only\n"
        "JAO context is plotted for inspection; current run logic uses JAO persona profile, "
        "not confirmed live CNEC-driven clearing.",
        fontsize=12,
    )
    jao = bundle.jao
    if jao is None or jao.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No JAO rows available for this run window", ha="center", va="center")
            ax.set_axis_off()
    else:
        axes[0].plot(
            jao["timestamp_utc"],
            jao["max_shadow_price_eur_mw"].fillna(0),
            color="#d62728",
            marker="o",
        )
        axes[0].set_ylabel("Max shadow price\nEUR/MW")
        axes[0].grid(alpha=0.25)
        axes[1].plot(jao["timestamp_utc"], jao["min_ram_mw"], color="#1f77b4", marker="o")
        axes[1].set_ylabel("Min RAM MW")
        axes[1].grid(alpha=0.25)
        axes[2].plot(jao["timestamp_utc"], jao["max_abs_flow_mw"], color="#ff7f0e", marker="o")
        axes[2].set_ylabel("Max abs flow MW")
        axes[2].grid(alpha=0.25)
        _format_time_axis(axes[2])
    _save(fig, bundle.output_dir / "05_jao_context.png")


def _plot_summary_comparison(bundles: list[RunBundle], output_path: Path) -> None:
    rows = []
    for bundle in bundles:
        summary = bundle.summary
        rows.append(
            {
                "run_id": bundle.run_id,
                "profit": summary.get("realized_profit_eur", 0),
                "must_watch_precision": summary.get("must_watch_profitable_or_activated_precision", 0),
                "must_watch_recall": summary.get("must_watch_profitable_or_activated_recall", 0),
                "bid_ticks": summary.get("bid_tick_count", 0),
                "watch_ticks": summary.get("watch_tick_count", 0),
                "wrong_side": summary.get("wrong_side_count", 0),
                "opportunity_capture": summary.get("opportunity_capture")
                or summary.get("truth_window_oracle_capture", 0),
            }
        )
    df = pd.DataFrame(rows)
    labels = [_short_run_label(run_id) for run_id in df["run_id"]]
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle("Recent good run comparison", fontsize=14)
    metrics = [
        ("profit", "Realized profit EUR"),
        ("must_watch_precision", "Must-watch precision"),
        ("must_watch_recall", "Must-watch recall"),
        ("bid_ticks", "Bid ticks"),
        ("watch_ticks", "Watch ticks"),
        ("wrong_side", "Wrong-side count"),
    ]
    for ax, (column, title) in zip(axes.flatten(), metrics, strict=True):
        ax.bar(labels, df[column], color="#1f77b4")
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, output_path)


def _shade_high_must_watch(ax: plt.Axes, ticks: pd.DataFrame) -> None:
    for timestamp in ticks.loc[ticks["high_must_watch"], "timestamp_utc"]:
        ax.axvspan(
            timestamp - pd.Timedelta(minutes=7.5),
            timestamp + pd.Timedelta(minutes=7.5),
            color="#d62728",
            alpha=0.09,
            linewidth=0,
        )


def _format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.tick_params(axis="x", labelrotation=25)


def _label(column: str) -> str:
    return column.replace("_eur_mwh", "").replace("_price", "").replace("_", " ")


def _short_run_label(run_id: str) -> str:
    label = run_id.replace("bme-", "").replace("bfa-", "").replace("bcast-mem-", "")
    return label.replace("apr", "\napr")


def _safe_name(value: str) -> str:
    return value.replace("/", "_")


def _timestamp_slug() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"plot was not written: {path}")


if __name__ == "__main__":
    main()
