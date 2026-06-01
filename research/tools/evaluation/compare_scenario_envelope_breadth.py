from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


CONFIG_ROOT = Path("ai-society/configs/scenario-envelope-breadth-20260520")
RUN_ROOT = Path("ai-society/runs/scenario-envelope-breadth-20260520")
OUTPUT_DIR = Path("evaluations/scenario-envelope-breadth-20260520")

METRICS = [
    "realized_profit_eur",
    "truth_window_oracle_capture",
    "wrong_side_count",
    "profitable_watch_or_bid_recall",
    "must_watch_profitable_or_activated_precision",
    "asset_backend_disagreement_rate",
    "asset_backend_scenario_envelope_false_positive_rate",
    "asset_backend_scenario_envelope_accepted_count",
    "asset_backend_pypsa_background_accepted_count",
    "asset_backend_proxy_accepted_count",
    "asset_backend_proxy_false_positive_count",
    "asset_backend_comparison_count",
    "unsupported_bid_proposal_rate",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CONFIG_ROOT / "manifest.json").read_text(encoding="utf-8"))

    scenario_rows = []
    pypsa_rows = []
    full_day_rows = []
    for item in manifest["full_runs"]:
        row = _row(item)
        if row["ticks"] == 96:
            full_day_rows.append(row)
        elif row["backend"] == "scenario":
            scenario_rows.append(row)
        elif row["backend"] == "pypsa":
            pypsa_rows.append(row)

    scenario_deltas = _scenario_deltas(scenario_rows)
    pypsa_deltas = _pypsa_deltas(pypsa_rows)

    payload = {
        "matrix": "scenario-envelope-breadth-20260520",
        "scenario_rows": scenario_rows,
        "scenario_medium_large_deltas": scenario_deltas,
        "pypsa_rows": pypsa_rows,
        "pypsa_tau_deltas": pypsa_deltas,
        "full_day_rows": full_day_rows,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(OUTPUT_DIR / "scenario_sizing_rows.csv", scenario_rows)
    _write_csv(OUTPUT_DIR / "scenario_sizing_deltas.csv", scenario_deltas)
    _write_csv(OUTPUT_DIR / "pypsa_tau_rows.csv", pypsa_rows)
    _write_csv(OUTPUT_DIR / "pypsa_tau_deltas.csv", pypsa_deltas)
    _write_csv(OUTPUT_DIR / "full_day_rows.csv", full_day_rows)
    _write_markdown(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _row(item: dict[str, Any]) -> dict[str, Any]:
    run_id = item["run_id"]
    summary = _evaluation_summary(run_id)
    run_summary = _run_summary(run_id)
    forecast_counts = Counter((run_summary.get("forecast_backend_by_agent") or {}).values())
    row = {
        "run_id": run_id,
        "society": _society(run_id),
        "window": _window(item),
        "ticks": item["ticks"],
        "backend": _backend(item["asset_simulator_mode"]),
        "sizing": item["candidate_sizing_mode"],
        "tau": item["verifier_tau_eur"],
        "forecast_backend_counts": dict(sorted(forecast_counts.items())),
    }
    for metric in METRICS:
        row[metric] = summary.get(metric)
    return row


def _scenario_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["society"], row["window"], row["sizing"]): row for row in rows}
    deltas = []
    for society in sorted({row["society"] for row in rows}):
        for window in sorted({row["window"] for row in rows}):
            medium = by_key.get((society, window, "medium"))
            large = by_key.get((society, window, "large"))
            if not medium or not large:
                continue
            row = {
                "society": society,
                "window": window,
                "left_run_id": large["run_id"],
                "right_run_id": medium["run_id"],
                "delta_label": "large_minus_medium",
            }
            for metric in METRICS:
                row[f"delta_{metric}"] = _delta(large.get(metric), medium.get(metric))
            deltas.append(row)
    return deltas


def _pypsa_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["window"], float(row["tau"])): row for row in rows}
    deltas = []
    for window in sorted({row["window"] for row in rows}):
        tau50 = by_key.get((window, -50.0))
        tau100 = by_key.get((window, -100.0))
        if not tau50 or not tau100:
            continue
        row = {
            "society": "s12-balanced",
            "window": window,
            "left_run_id": tau100["run_id"],
            "right_run_id": tau50["run_id"],
            "delta_label": "tau_minus_100_minus_tau_minus_50",
        }
        for metric in METRICS:
            row[f"delta_{metric}"] = _delta(tau100.get(metric), tau50.get(metric))
        deltas.append(row)
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


def _society(run_id: str) -> str:
    if run_id.startswith("seb-s12-balanced-"):
        return "s12-balanced"
    if run_id.startswith("seb-s20-mixed-persona-"):
        return "s20-mixed-persona"
    raise RuntimeError(f"cannot infer society from run_id: {run_id}")


def _backend(mode: str) -> str:
    if mode == "dual_compare_real_controls":
        return "scenario"
    if mode == "dual_compare_pypsa_controls":
        return "pypsa"
    raise RuntimeError(f"unexpected asset simulator mode: {mode}")


def _window(item: dict[str, Any]) -> str:
    windows = {
        "2026-04-04T06:00:00Z": "apr04-0600",
        "2026-04-17T19:00:00Z": "apr17-1900",
        "2026-04-19T19:15:00Z": "apr19-1915",
        "2026-04-28T19:00:00Z": "apr28-1900",
        "2026-04-17T00:00:00Z": "apr17-0000",
        "2026-04-28T00:00:00Z": "apr28-0000",
    }
    timestamp = item["start_timestamp"]
    try:
        return windows[timestamp]
    except KeyError as exc:
        raise RuntimeError(f"unexpected window timestamp: {timestamp}") from exc


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
    lines = [
        "# Scenario Envelope Breadth 2026-05-20",
        "",
        "## Scenario Medium vs Large",
        "",
        "| society | window | profit delta | wrong-side delta | false-positive delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["scenario_medium_large_deltas"]:
        lines.append(
            "| {society} | {window} | {profit} | {wrong} | {fp} |".format(
                society=row["society"],
                window=row["window"],
                profit=_fmt(row.get("delta_realized_profit_eur")),
                wrong=_fmt(row.get("delta_wrong_side_count")),
                fp=_fmt(row.get("delta_asset_backend_scenario_envelope_false_positive_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## PyPSA Tau -100 vs -50",
            "",
            "| window | profit delta | wrong-side delta | false-positive delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["pypsa_tau_deltas"]:
        lines.append(
            "| {window} | {profit} | {wrong} | {fp} |".format(
                window=row["window"],
                profit=_fmt(row.get("delta_realized_profit_eur")),
                wrong=_fmt(row.get("delta_wrong_side_count")),
                fp=_fmt(row.get("delta_asset_backend_scenario_envelope_false_positive_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## Full-Day Examples",
            "",
            "| run | society | window | sizing | profit | wrong-side | forecast backends |",
            "|---|---|---:|---|---:|---:|---|",
        ]
    )
    for row in payload["full_day_rows"]:
        lines.append(
            "| {run_id} | {society} | {window} | {sizing} | {profit} | {wrong} | `{forecasters}` |".format(
                run_id=row["run_id"],
                society=row["society"],
                window=row["window"],
                sizing=row["sizing"],
                profit=_fmt(row.get("realized_profit_eur")),
                wrong=_fmt(row.get("wrong_side_count")),
                forecasters=json.dumps(row.get("forecast_backend_counts", {}), sort_keys=True),
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
