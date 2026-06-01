"""Compare opportunity capture + safety across the llm-agency-capture matrix.

Reads evaluations/<run_id>/run_summary.json for every run in the matrix manifest and
emits a mode x window table of the primary axis (does grounded LLM agency close the
~5-16% capture gap?) and the safety axis (does the verifier floor still hold?).

Usage:
  PYTHONPATH=. uv run python tools/evaluation/compare_agency_capture.py \
    [--manifest ai-society/configs/llm-agency-capture-20260522/manifest.json] \
    [--evaluations-dir evaluations]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("ai-society/configs/llm-agency-capture-20260522/manifest.json")

# Agency order for stable, readable sorting.
MODE_ORDER = ["a0-det", "a1-selector", "a2pure-cp11", "a2hyb-cp12", "a3-tau100", "a3-tau250", "a4-cp13-refine"]

# Primary (capture) + safety + decomposition metrics pulled from run_summary.json.
METRICS = [
    "opportunity_capture",
    "truth_window_oracle_capture",
    "realized_profit_eur",
    "truth_window_oracle_profit_eur",
    "regret_eur",
    "downside_cvar_95_eur",
    "max_drawdown_eur",
    "verifier_false_accepts",
    "verifier_realized_profit_breach_rate",
    "profitable_bid_recall",
    "wrong_side_count",
    "bid_action_count",
    "filled_count",
    "autonomous_tool_call_rate",
    "unsupported_bid_proposal_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare agency-capture matrix runs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--evaluations-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    matrix = manifest["matrix"]
    out_dir = args.out_dir or (args.evaluations_dir / f"{matrix}-compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    pending: list[str] = []
    for item in manifest["full_runs"]:
        run_id = item["run_id"]
        mode = _mode_of(run_id)
        window = _window_of(item["start_timestamp"])
        summary_path = args.evaluations_dir / run_id / "run_summary.json"
        if not summary_path.exists():
            pending.append(run_id)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {"run_id": run_id, "mode": mode, "window": window, "verifier_tau_eur": item.get("verifier_tau_eur")}
        for metric in METRICS:
            row[metric] = summary.get(metric)
        rows.append(row)

    rows.sort(key=lambda r: (MODE_ORDER.index(r["mode"]) if r["mode"] in MODE_ORDER else 99, r["window"]))
    mode_means = _mode_means(rows)

    payload = {
        "matrix": matrix,
        "evaluated_runs": len(rows),
        "pending_runs": pending,
        "rows": rows,
        "mode_means": mode_means,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "rows.csv", rows, ["run_id", "mode", "window", "verifier_tau_eur", *METRICS])
    _write_csv(out_dir / "mode_means.csv", mode_means, ["mode", "n", "opportunity_capture", "realized_profit_eur", "downside_cvar_95_eur", "verifier_false_accepts", "verifier_realized_profit_breach_rate", "profitable_bid_recall", "wrong_side_count"])
    _print_table(rows, mode_means, pending)


def _mode_of(run_id: str) -> str:
    # lac-s06-<mode>-<window>-seed...; mode may contain hyphens, window is aprNN-HHMM.
    body = run_id.removeprefix("lac-s06-")
    parts = body.split("-")
    # window token is like apr02-0530 -> two tokens; mode is everything before it.
    for idx in range(len(parts)):
        if parts[idx].startswith("apr"):
            return "-".join(parts[:idx])
    return body


def _window_of(start_timestamp: str) -> str:
    # 2026-04-02T05:30:00Z -> apr02-0530
    date, _, time = start_timestamp.partition("T")
    month_day = date[5:10].replace("-", "")
    months = {"04": "apr", "05": "may", "03": "mar"}
    mm = months.get(month_day[:2], month_day[:2])
    return f"{mm}{month_day[2:]}-{time[0:2]}{time[3:5]}"


def _mode_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    out = []
    for mode in sorted(by_mode, key=lambda m: MODE_ORDER.index(m) if m in MODE_ORDER else 99):
        group = by_mode[mode]
        out.append(
            {
                "mode": mode,
                "n": len(group),
                "opportunity_capture": _mean(group, "opportunity_capture"),
                "realized_profit_eur": _mean(group, "realized_profit_eur"),
                "downside_cvar_95_eur": _mean(group, "downside_cvar_95_eur"),
                "verifier_false_accepts": _sum(group, "verifier_false_accepts"),
                "verifier_realized_profit_breach_rate": _mean(group, "verifier_realized_profit_breach_rate"),
                "profitable_bid_recall": _mean(group, "profitable_bid_recall"),
                "wrong_side_count": _sum(group, "wrong_side_count"),
            }
        )
    return out


def _mean(group: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in group if isinstance(r.get(key), (int, float))]
    return round(sum(values) / len(values), 6) if values else None


def _sum(group: list[dict[str, Any]], key: str) -> float:
    return round(sum(float(r[key]) for r in group if isinstance(r.get(key), (int, float))), 6)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _print_table(rows: list[dict[str, Any]], mode_means: list[dict[str, Any]], pending: list[str]) -> None:
    print("\n=== per-mode means (agency order) ===")
    cols = ["mode", "n", "opportunity_capture", "realized_profit_eur", "downside_cvar_95_eur", "verifier_false_accepts", "profitable_bid_recall", "wrong_side_count"]
    print(" | ".join(f"{c:>22}" if c == "mode" else f"{c:>16}" for c in cols))
    for m in mode_means:
        print(" | ".join(f"{_fmt(m.get(c)):>22}" if c == "mode" else f"{_fmt(m.get(c)):>16}" for c in cols))
    print(f"\nevaluated={len(rows)} pending={len(pending)}")
    if pending:
        print("pending run_ids:")
        for run_id in pending:
            print(f"  {run_id}")


if __name__ == "__main__":
    main()
