from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


KEYS = [
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
]


def main() -> None:
    rows = []
    for path in Path("evaluations").glob("mi-*/run_summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        run_id = summary.get("run_id") or path.parent.name
        parsed = _parse_run_id(run_id)
        if parsed is None:
            continue
        rows.append({**parsed, **{key: summary.get(key) for key in KEYS}})
    out_dir = Path("ai-society/runs/market-intelligence-full-suite")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "suite-results.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    grouped = {
        "by_size": _group(rows, ["size"]),
        "by_comm": _group(rows, ["comm"]),
        "by_forecaster": _group(rows, ["forecaster"]),
        "by_size_comm": _group(rows, ["size", "comm"]),
        "by_window_kind": _group(rows, ["window_kind"]),
    }
    payload = {"run_count": len(rows), "csv": str(out_csv), **grouped}
    (out_dir / "suite-summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parse_run_id(run_id: str) -> dict[str, str | int] | None:
    parts = run_id.split("-")
    if len(parts) < 7 or parts[0] != "mi" or parts[-1] != "q32":
        return None
    forecaster = parts[-2]
    comm = parts[3]
    if comm not in {"independent", "broadcast", "peer", "retry"}:
        return None
    window = "-".join(parts[4:-2])
    return {
        "run_id": run_id,
        "size": int(parts[1]) if parts[1].isdigit() else parts[1],
        "profile": parts[2],
        "comm": comm,
        "window": window,
        "window_kind": "rolling" if window.startswith("roll") else "known",
        "forecaster": forecaster,
    }


def _group(rows: list[dict], keys: list[str]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[key] for key in keys), []).append(row)
    out = []
    for values, group_rows in sorted(buckets.items(), key=lambda item: item[0]):
        payload = {key: value for key, value in zip(keys, values, strict=True)}
        payload["count"] = len(group_rows)
        for metric in KEYS:
            vals = [row[metric] for row in group_rows if isinstance(row.get(metric), int | float)]
            payload[f"avg_{metric}"] = round(mean(vals), 6) if vals else None
        out.append(payload)
    return out


if __name__ == "__main__":
    main()
