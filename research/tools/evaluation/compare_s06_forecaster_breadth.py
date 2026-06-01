from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CONFIG_ROOT = Path("ai-society/configs/s06-forecaster-breadth-20260520")
RUN_ROOT = Path("ai-society/runs/s06-forecaster-breadth-20260520")
OUTPUT_DIR = Path("evaluations/s06-forecaster-breadth-20260520")

METRICS = [
    "realized_profit_eur",
    "truth_window_oracle_capture",
    "wrong_side_count",
    "profitable_watch_or_bid_recall",
    "must_watch_profitable_or_activated_precision",
    "unsupported_bid_proposal_rate",
    "evidence_reference_validity",
]
DIAGNOSTIC_COUNT_KEYS = [
    "accepted_simulator_candidate_counts",
    "rejected_simulator_candidate_counts",
    "agent_forecast_backend_counts",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CONFIG_ROOT / "manifest.json").read_text(encoding="utf-8"))

    rows = [_row(item) for item in manifest["full_runs"]]
    deltas = _window_deltas(rows)
    payload = {
        "matrix": "s06-forecaster-breadth-20260520",
        "forecaster_validation": manifest.get("forecaster_validation", {}),
        "rows": rows,
        "forecaster_window_deltas": deltas,
        "notes": _notes(manifest),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(OUTPUT_DIR / "forecaster_rows.csv", rows)
    _write_csv(OUTPUT_DIR / "forecaster_window_deltas.csv", deltas)
    _write_markdown(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _row(item: dict[str, Any]) -> dict[str, Any]:
    run_id = item["run_id"]
    summary = _evaluation_summary(run_id)
    run_summary = _run_summary(run_id)
    side = run_summary.get("side_diagnostics") or {}
    row = {
        "run_id": run_id,
        "window": _window(item["start_timestamp"]),
        "ticks": item["ticks"],
        "intended_forecaster": item["intended_forecaster"],
        "actual_forecaster": item["actual_forecaster"],
        "fallback_used": item["fallback_used"],
        "fallback_reason": item["fallback_reason"],
        "internal_fallback_note": item["internal_fallback_note"],
    }
    for metric in METRICS:
        row[metric] = summary.get(metric)
    for key in DIAGNOSTIC_COUNT_KEYS:
        row[key] = side.get(key, {})
    return row


def _window_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_window = {
        row["window"]: row
        for row in rows
        if row["intended_forecaster"] == "f8" and not row["fallback_used"]
    }
    deltas = []
    for row in sorted(rows, key=lambda item: (item["window"], item["intended_forecaster"])):
        baseline = baseline_by_window.get(row["window"])
        if not baseline:
            continue
        delta = {
            "window": row["window"],
            "intended_forecaster": row["intended_forecaster"],
            "actual_forecaster": row["actual_forecaster"],
            "baseline_forecaster": "f8",
            "run_id": row["run_id"],
            "baseline_run_id": baseline["run_id"],
            "fallback_used": row["fallback_used"],
        }
        for metric in METRICS:
            delta[f"delta_vs_f8_{metric}"] = _delta(row.get(metric), baseline.get(metric))
        deltas.append(delta)
    return deltas


def _evaluation_summary(run_id: str) -> dict[str, Any]:
    path = Path("evaluations") / run_id / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid evaluation summary: {path}")
    return payload


def _run_summary(run_id: str) -> dict[str, Any]:
    path = RUN_ROOT / run_id / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid run summary: {path}")
    return payload


def _window(timestamp: str) -> str:
    windows = {
        "2026-04-02T05:30:00Z": "apr02-0530",
        "2026-04-09T18:30:00Z": "apr09-1830",
        "2026-04-13T00:15:00Z": "apr13-0015",
    }
    try:
        return windows[timestamp]
    except KeyError as exc:
        raise RuntimeError(f"unexpected window timestamp: {timestamp}") from exc


def _notes(manifest: dict[str, Any]) -> list[str]:
    notes = []
    validation = manifest.get("forecaster_validation", {})
    for name, record in sorted(validation.items()):
        if record.get("fallback_used"):
            notes.append(
                f"{name} used configured fallback {record.get('actual_forecaster')}: {record.get('fallback_reason')}"
            )
        if record.get("internal_fallback_note"):
            notes.append(str(record["internal_fallback_note"]))
    return notes


def _delta(left: Any, right: Any) -> float | None:
    try:
        return round(float(left) - float(right), 6)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_markdown(payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# S06 Forecaster Breadth 2026-05-20",
        "",
        "## Fallbacks",
        "",
    ]
    notes = payload["notes"]
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- No configured backend fallbacks were used.")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| forecaster | actual | window | fallback | profit | wrong-side | recall | must-watch precision | evidence |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["intended_forecaster"], item["window"])):
        lines.append(
            "| {intended} | {actual} | {window} | {fallback} | {profit} | {wrong} | {recall} | {precision} | {evidence} |".format(
                intended=row["intended_forecaster"],
                actual=row["actual_forecaster"],
                window=row["window"],
                fallback="yes" if row["fallback_used"] else "no",
                profit=_fmt(row.get("realized_profit_eur")),
                wrong=_fmt(row.get("wrong_side_count")),
                recall=_fmt(row.get("profitable_watch_or_bid_recall")),
                precision=_fmt(row.get("must_watch_profitable_or_activated_precision")),
                evidence=_fmt(row.get("evidence_reference_validity")),
            )
        )

    lines.extend(
        [
            "",
            "## Delta vs F8",
            "",
            "| forecaster | window | profit delta | wrong-side delta | recall delta | precision delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["forecaster_window_deltas"]:
        lines.append(
            "| {forecaster} | {window} | {profit} | {wrong} | {recall} | {precision} |".format(
                forecaster=row["intended_forecaster"],
                window=row["window"],
                profit=_fmt(row.get("delta_vs_f8_realized_profit_eur")),
                wrong=_fmt(row.get("delta_vs_f8_wrong_side_count")),
                recall=_fmt(row.get("delta_vs_f8_profitable_watch_or_bid_recall")),
                precision=_fmt(row.get("delta_vs_f8_must_watch_profitable_or_activated_precision")),
            )
        )

    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
