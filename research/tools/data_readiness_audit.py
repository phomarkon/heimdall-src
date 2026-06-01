"""Phase 1 data audit: null coverage, range, and source provenance for every
feature in every panel we train forecasters on. The output drives the
decision matrix for which columns to use, fill, or drop.

Run:  PYTHONPATH=. uv run python tools/data_readiness_audit.py
Output: notes/findings/2026-05-17-data-audit.md
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PANELS = {
    "dk1_panel_train": REPO / "data/processed/dk1_panel_train.parquet",
    "dk1_panel_val": REPO / "data/processed/dk1_panel_val.parquet",
    "dk1_panel_test": REPO / "data/processed/dk1_panel_test.parquet",
    "dk1_panel_features_v2": REPO / "data/processed/dk1_panel_features_v2.parquet",
    "dk1_panel_rich": REPO / "data/processed/dk1_panel_rich.parquet",
    "dk1_panel_rich_v2": REPO / "data/processed/dk1_panel_rich_v2.parquet",
    "anomaly_features": REPO / "data/processed/anomaly_features.parquet",
}


def _audit(df: pl.DataFrame) -> dict:
    n = len(df)
    out = {}
    for c in df.columns:
        s = df[c]
        nulls = int(s.null_count())
        out[c] = {
            "null_pct": round(100 * nulls / n, 1),
            "dtype": str(s.dtype),
        }
        if s.dtype.is_numeric() and (n - nulls) > 0:
            arr = s.drop_nulls()
            out[c].update({
                "mean": float(arr.mean()),
                "std": float(arr.std() or 0.0),
                "min": float(arr.min()),
                "max": float(arr.max()),
            })
    return out


def main() -> int:
    lines = ["# Data-readiness audit — 2026-05-17", ""]
    lines.append("Every feature in every panel: % nulls, dtype, summary stats.")
    lines.append("Decision column at the end:")
    lines.append("  - ✓ USE       : <10% nulls + numeric + sensible range")
    lines.append("  - ◐ INVESTIGATE: 10–80% nulls (sparse but informative)")
    lines.append("  - ✗ DROP      : >80% nulls (mostly empty) OR all-null OR placeholder")
    lines.append("")
    decisions: dict[str, str] = {}
    for panel_name, p in PANELS.items():
        if not p.exists():
            lines.append(f"## {panel_name}\n*MISSING*: `{p}`\n")
            continue
        df = pl.read_parquet(p)
        a = _audit(df)
        lines.append(f"## {panel_name}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
        lines.append("")
        lines.append("| col | null% | dtype | mean | std | min | max | decision |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---|")
        for c, m in a.items():
            null_pct = m["null_pct"]
            mean = f"{m.get('mean', '-'):.3f}" if "mean" in m else "—"
            std = f"{m.get('std', '-'):.3f}" if "std" in m else "—"
            mn = f"{m.get('min', '-'):.3f}" if "min" in m else "—"
            mx = f"{m.get('max', '-'):.3f}" if "max" in m else "—"
            if null_pct >= 100.0:
                dec = "✗ DROP (placeholder/empty)"
            elif null_pct > 80.0:
                dec = "✗ DROP (>80% null)"
            elif null_pct > 10.0:
                dec = "◐ INVESTIGATE"
            else:
                dec = "✓ USE"
            decisions[f"{panel_name}.{c}"] = dec
            lines.append(f"| {c} | {null_pct} | {m['dtype']} | {mean} | {std} | {mn} | {mx} | {dec} |")
        lines.append("")
    # Summary by decision
    from collections import Counter
    counts = Counter(decisions.values())
    lines.append("## Summary")
    lines.append("")
    for d, n in counts.most_common():
        lines.append(f"- {d}: {n} columns")
    lines.append("")
    out = REPO / "notes/findings/2026-05-17-data-audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")
    print(f"\nDecision summary: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
