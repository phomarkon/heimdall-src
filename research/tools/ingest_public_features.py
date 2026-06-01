"""Public-data feature ingest (free-tier only).

Per docs/RESEARCH-PROPOSAL.md §5.1 + the 2026-05-10 strategy session.  Pulls
free, no-auth public datasets and joins them onto the DK1 panel.  The
output panel adds wind-forecast / solar-forecast / actual-generation /
cross-border-flow / carbon-intensity features that the existing 7-column
panel lacks; these are the highest-leverage features for the panel.

Sources (all free, all no-auth):
  - Energinet Energi Data Service — Forecasts_Hour, ProductionConsumptionSettlement,
    Elspotprices, CO2Emis.  REST/JSON, no API key.
  - Open Power System Data — fallback / cross-validation for older windows.

Output:
  - `data/processed/dk1_panel_features_v2.parquet` — augmented panel
    (15-min ticks, joined by 1-h forward-fill on the hourly Energinet
    columns).
  - `data/processed/feature_provenance.json` — per-source SHA-256
    + fetch timestamp.

Usage:
  PYTHONPATH=. python tools/ingest_public_features.py \\
      --start 2023-05-22T00:00 --end 2026-04-29T23:45 --zone DK1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data/raw/public_features"


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:16]


def _save_raw(name: str, rows: list[dict]) -> tuple[Path, str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.json"
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(payload)
    return path, _hash(payload)


def fetch_energinet(start: str, end: str, zone: str = "DK1") -> dict:
    from heimdall_data.energinet import EnerginetClient
    cli = EnerginetClient()
    out = {}
    print(f"[energinet] Forecasts_Hour {zone} {start}..{end}")
    out["forecasts_hour"] = cli.forecasts_hour(start, end, zone)
    print(f"[energinet] ProductionConsumptionSettlement {zone}")
    out["production"] = cli.production_consumption_settlement(start, end, zone)
    print(f"[energinet] Elspotprices {zone}")
    out["elspot"] = cli.elspot_prices(start, end, zone)
    print(f"[energinet] CO2Emis {zone}")
    out["co2"] = cli.co2_emis(start, end, zone)
    return out


def _to_pandas(rows: list[dict], time_col: str = "HourUTC") -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if time_col not in df.columns:
        # Some datasets use ``Minutes5UTC``; remap.
        for alt in ("Minutes5UTC", "MinutesUTC", "TimestampUTC"):
            if alt in df.columns:
                df = df.rename(columns={alt: time_col})
                break
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    return df


def build_extended_panel(start: str, end: str, zone: str = "DK1") -> dict:
    raw = fetch_energinet(start, end, zone)
    provenance = {}
    for name, rows in raw.items():
        path, sha = _save_raw(f"{zone.lower()}_{name}", rows)
        provenance[name] = {
            "source": "energinet",
            "rows": len(rows),
            "sha256": sha,
            "path": str(path.relative_to(REPO_ROOT)),
            "fetched_at_utc": datetime.utcnow().isoformat() + "Z",
        }

    # Wide-pivot Forecasts_Hour: rows are long (one per ForecastType per hour).
    fc = _to_pandas(raw["forecasts_hour"])
    if not fc.empty and "ForecastType" in fc.columns:
        fc = fc.pivot_table(
            index="HourUTC",
            columns="ForecastType",
            values="ForecastDayAhead",
            aggfunc="first",
        ).reset_index()
        fc.columns = ["HourUTC"] + [
            f"forecast_da_mw_{c.lower().replace(' ', '_')}" for c in fc.columns[1:]
        ]
    prod = _to_pandas(raw["production"])
    elspot = _to_pandas(raw["elspot"])
    co2 = _to_pandas(raw["co2"], time_col="Minutes5UTC")
    if not co2.empty and "Minutes5UTC" not in co2.columns:
        # already renamed by _to_pandas to HourUTC; rename back for hourly join
        co2 = co2.rename(columns={"HourUTC": "Minutes5UTC"})

    # Build the 15-min spine from the existing DK1 panel.
    base = pl.read_parquet(REPO_ROOT / "data/processed/dk1_panel.parquet").sort("timestamp_utc")
    spine = base.select(
        pl.col("timestamp_utc"),
    )
    spine_pd = spine.to_pandas()
    spine_pd["timestamp_utc"] = pd.to_datetime(spine_pd["timestamp_utc"], utc=True).astype("datetime64[ns, UTC]")

    # Forward-fill 1-h hourly features onto 15-min spine.
    for src_df, label in [(fc, "fc"), (prod, "prod"), (elspot, "elspot")]:
        if src_df.empty:
            continue
        src_df = src_df.set_index("HourUTC").sort_index()
        spine_pd = pd.merge_asof(
            spine_pd.sort_values("timestamp_utc"),
            src_df.reset_index().rename(columns={"HourUTC": "timestamp_utc"}).sort_values("timestamp_utc"),
            on="timestamp_utc",
            direction="backward",
            tolerance=pd.Timedelta("1h"),
        )
    if not co2.empty:
        co2 = co2.set_index("Minutes5UTC").sort_index().reset_index().rename(
            columns={"Minutes5UTC": "timestamp_utc"}
        )
        spine_pd = pd.merge_asof(
            spine_pd.sort_values("timestamp_utc"),
            co2[["timestamp_utc", "CO2Emission"]].sort_values("timestamp_utc"),
            on="timestamp_utc",
            direction="backward",
            tolerance=pd.Timedelta("15min"),
        )

    # Coerce numeric columns.
    for c in spine_pd.columns:
        if c == "timestamp_utc":
            continue
        spine_pd[c] = pd.to_numeric(spine_pd[c], errors="coerce")

    out_path = REPO_ROOT / "data/processed/dk1_panel_features_v2.parquet"
    pl.from_pandas(spine_pd).write_parquet(out_path)

    prov_path = REPO_ROOT / "data/processed/feature_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))

    return {
        "panel_path": str(out_path.relative_to(REPO_ROOT)),
        "provenance_path": str(prov_path.relative_to(REPO_ROOT)),
        "n_rows": int(spine_pd.shape[0]),
        "n_cols": int(spine_pd.shape[1]),
        "columns": list(spine_pd.columns),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-05-22T00:00")
    p.add_argument("--end", default="2026-04-29T23:45")
    p.add_argument("--zone", default="DK1")
    args = p.parse_args()
    t0 = time.time()
    res = build_extended_panel(args.start, args.end, args.zone)
    res["wall_seconds"] = round(time.time() - t0, 2)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
