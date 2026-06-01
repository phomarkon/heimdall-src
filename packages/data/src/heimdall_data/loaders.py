"""Tidy panel loader for DK1.

Returns a polars DataFrame aligned to a 15-min UTC index over [start, end) with
the columns the forecaster zoo (docs/RESEARCH-PROPOSAL.md §4.2.2) consumes:

  - ``imbalance_price_dkk_mwh``  (Energinet RegulatingBalancePowerdata)
  - ``mfrr_up_volume_mw``        (Energinet mFRREnergyActivated, direction up)
  - ``mfrr_down_volume_mw``      (Energinet mFRREnergyActivated, direction down)
  - ``da_price_dkk_mwh``         (ENTSO-E A44, day-ahead)
    ⚠️ **Unit caveat (filed 2026-05-16):** ENTSO-E A44 publishes DK1 prices in
    **EUR/MWh**, not DKK. We kept the historical column name to avoid a
    cascading rename across the trainer / verifier / paper, but the units in
    this column are **EUR/MWh**. The imbalance columns *are* genuine DKK.
    When regenerating the schema, rename to ``da_price_eur_mwh``.
  - ``load_actual_mw``           (ENTSO-E A65, total load actual)

Per the proposal §5.7, the train/val/test split is deterministic at
2025-03-04 00:00 UTC. The loader does not enforce a split — that is the
caller's responsibility (see ``apps/forecaster/train``) — but it accepts a
``split`` hint that simply records which side of the break was requested.

Persistence:
  - raw responses cached under ``data/raw/{source}/{yyyy-mm}/*.parquet``
    (DVC-tracked once we wire DVC),
  - tidy panel under ``data/processed/dk1_panel.parquet``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl

from heimdall_data.energinet import EnerginetClient
from heimdall_data.entsoe import EntsoeClient

REPO_ROOT = Path(__file__).resolve().parents[4]
RAW_ROOT = REPO_ROOT / "data" / "raw"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
PRE_POST_BREAK_UTC = datetime(2025, 3, 4, 0, 0, tzinfo=timezone.utc)


SplitHint = Literal["pre", "post", "all"]


def _quarter_index(start: datetime, end: datetime) -> pl.Series:
    """15-minute UTC index ∈ [start, end). Typed datetime[μs, UTC]."""
    rng = pd.date_range(start=start, end=end, freq="15min", tz="UTC", inclusive="left")
    return pl.Series("timestamp_utc", rng.to_pydatetime().tolist()).cast(
        pl.Datetime("us", time_zone="UTC")
    )


def _to_polars(s: pd.Series | pd.DataFrame, name: str) -> pl.DataFrame:
    """Convert a pandas time-series to a 2-col polars frame on the 15-min grid.

    Forward-fills sub-15-min values. Resampling to 15-min uses 'first' to be
    deterministic; for prices that's a fine choice given Nordic market mechanics.
    """
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.tz_convert("UTC")
    g = s.resample("15min").first().rename(name)
    ts = pl.Series("timestamp_utc", g.index.to_pydatetime().tolist()).cast(
        pl.Datetime("us", time_zone="UTC")
    )
    vals = pl.Series(name, g.to_numpy(dtype="float64"), dtype=pl.Float64)
    return pl.DataFrame([ts, vals])


def load_dk1_panel(
    start: datetime,
    end: datetime,
    *,
    split: SplitHint = "all",
    entsoe: EntsoeClient | None = None,
    energinet: EnerginetClient | None = None,
    cache_dir: Path | None = None,
) -> pl.DataFrame:
    """Build the DK1 tidy panel.

    Live API calls are made only when the caller does not pass pre-built clients
    (or stub responses) — tests mock both clients to avoid network access.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start/end must be tz-aware (UTC).")
    if split == "pre" and end > PRE_POST_BREAK_UTC:
        raise ValueError("split='pre' but end is after 2025-03-04; use split='all'")
    if split == "post" and start < PRE_POST_BREAK_UTC:
        raise ValueError("split='post' but start is before 2025-03-04; use split='all'")

    cache_dir = cache_dir or PROCESSED_ROOT
    entsoe = entsoe or EntsoeClient()
    energinet = energinet or EnerginetClient()

    pd_start = pd.Timestamp(start)
    pd_end = pd.Timestamp(end)

    # -- ENTSO-E pulls -----------------------------------------------------
    da = entsoe.day_ahead_prices(pd_start, pd_end)
    load = entsoe.total_load(pd_start, pd_end, kind="actual")
    # ENTSO-E A85 returns 15-min granularity post-2025-03-04 (probed live
    # 2026-05-09). Pre-break it falls back to hourly. We carry the "Long"
    # imbalance column (single-price publication, the standard DK1 metric).
    try:
        imb = entsoe.imbalance_prices(pd_start, pd_end)
    except Exception:  # noqa: BLE001 -- pre-break may 404; not a hard fail
        imb = None

    # -- Energinet pulls ---------------------------------------------------
    iso_start = pd_start.strftime("%Y-%m-%dT%H:%M")
    iso_end = pd_end.strftime("%Y-%m-%dT%H:%M")
    rb_rows = energinet.regulating_balance(iso_start, iso_end, area="DK1")

    # -- Build base 15-min grid -------------------------------------------
    base = pl.DataFrame({"timestamp_utc": _quarter_index(start, end)})

    # ENTSO-E series → polars
    da_pl = _to_polars(da, "da_price_dkk_mwh")
    load_pl = _to_polars(load, "load_actual_mw")

    # Energinet RegulatingBalancePowerdata carries imbalance price + mFRR
    # activated volumes in a single response. Schema probed 2026-05-09.
    # DEVIATION: Energinet's open API exposes this dataset at *hourly*
    # granularity even on post-2025-03-04 data; finer 15-min granularity is
    # only available via ENTSO-E A85. We forward-fill the hourly Energinet
    # signal onto the 15-min grid below — acceptable for v1; KE1 already
    # uses ENTSO-E A85 directly when finer data are required.
    rb_pl = _energinet_rows_to_polars(
        rb_rows,
        time_col="HourUTC",
        value_cols=(
            "ImbalancePriceDKK",
            "mFRRUpActBal",
            "mFRRDownActBal",
        ),
        rename_map={
            "ImbalancePriceDKK": "imbalance_price_dkk_mwh",
            "mFRRUpActBal": "mfrr_up_volume_mw",
            "mFRRDownActBal": "mfrr_down_volume_mw",
        },
    )

    frames = [da_pl, load_pl, rb_pl]
    if imb is not None:
        # ENTSO-E A85 "Long" imbalance price is the canonical 15-min signal
        # post-2025-03-04. We carry both EUR (raw) and DKK (×7.46 — fixed FX,
        # to be replaced with daily FX from Energinet for the final paper).
        long_col = imb["Long"] if "Long" in imb.columns else imb.iloc[:, 0]
        imb_eur = _to_polars(long_col, "imbalance_price_eur_mwh")
        imb_dkk_15min = imb_eur.with_columns(
            (pl.col("imbalance_price_eur_mwh") * 7.46).alias("imbalance_price_dkk_mwh_15min")
        ).select(["timestamp_utc", "imbalance_price_dkk_mwh_15min"])
        frames.extend([imb_eur, imb_dkk_15min])

    panel = base
    for frame in frames:
        panel = panel.join(frame, on="timestamp_utc", how="left")

    panel = panel.sort("timestamp_utc")
    # Forward-fill hourly columns onto the 15-min grid: Energinet columns,
    # ENTSO-E DA price (hourly), and ENTSO-E load (hourly). ENTSO-E A85
    # imbalance is already 15-min and needs no fill.
    fill_cols = [
        c
        for c in (
            "imbalance_price_dkk_mwh",
            "mfrr_up_volume_mw",
            "mfrr_down_volume_mw",
            "da_price_dkk_mwh",
            "load_actual_mw",
            "imbalance_price_eur_mwh",
            "imbalance_price_dkk_mwh_15min",
        )
        if c in panel.columns
    ]
    if fill_cols:
        panel = panel.with_columns(
            [
                # Convert NaN floats to nulls before forward-fill (polars
                # distinguishes the two but our hourly→15-min reindex creates
                # NaN floats, not nulls).
                pl.when(pl.col(c).is_nan())
                .then(None)
                .otherwise(pl.col(c))
                .forward_fill()
                .alias(c)
                for c in fill_cols
            ]
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"dk1_panel_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    panel.write_parquet(out_path)
    return panel


def _energinet_rows_to_polars(
    rows: list[dict],
    *,
    time_col: str,
    value_cols: tuple[str, ...],
    rename_map: dict[str, str],
) -> pl.DataFrame:
    """Normalise an Energinet JSON list into a polars 15-min frame."""
    if not rows:
        # Empty frame with the correct schema so the join still works.
        return pl.DataFrame(
            {
                "timestamp_utc": pl.Series([], dtype=pl.Datetime("us", time_zone="UTC")),
                **{rename_map[c]: pl.Series([], dtype=pl.Float64) for c in value_cols},
            }
        )
    df = pl.from_dicts(rows)
    keep = [time_col] + [c for c in value_cols if c in df.columns]
    df = df.select(keep)
    df = df.rename({time_col: "timestamp_utc"} | {c: rename_map[c] for c in value_cols if c in df.columns})
    df = df.with_columns(
        pl.col("timestamp_utc")
        .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S", strict=False)
        .dt.replace_time_zone("UTC")
    )
    # Cast numeric value columns
    for v in (rename_map[c] for c in value_cols if c in df.columns):
        df = df.with_columns(pl.col(v).cast(pl.Float64, strict=False))
    return df


__all__ = ["PRE_POST_BREAK_UTC", "PROCESSED_ROOT", "RAW_ROOT", "load_dk1_panel"]
