"""Rich panel builder: base DK1 panel + Trinity wide features + markets + calendar.

This is the data substrate for F8b / F8c / F8d / F12 (Plan v2 Tracks A.10, C, D).
The function ``build_rich_panel`` is designed to be runnable WITHOUT network — it
consumes the cached canonical ``data/processed/dk1_panel.parquet`` produced by
the original ``load_dk1_panel``, joins Trinity vendored features (24h-lagged
weather by default — see ``trinity.DEFAULT_WEATHER_LAG_HOURS``), joins fuel /
carbon markets (no-op if files absent), and adds calendar features.

Caching:
  - Writes to ``data/processed/dk1_panel_rich.parquet`` (and ``_archive`` variant
    if ``weather_lag_hours == 0``).
  - Use ``rebuild=True`` to force a recomputation.

Schema invariants:
  - 15-min UTC grid (matching the base panel).
  - Existing base columns are preserved with their original names and dtypes.
  - All new columns are clearly prefixed (``trinity_*``, ``flow_*``, ``umm_*``,
    fuel/carbon keys, calendar keys) so feature-selection scripts can grep them.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from heimdall_data.calendar_features import add_calendar_features
from heimdall_data.loaders import PROCESSED_ROOT
from heimdall_data.markets import load_markets
from heimdall_data.trinity import DEFAULT_WEATHER_LAG_HOURS, make_dk1_wide_features

BASE_PANEL_PATH = PROCESSED_ROOT / "dk1_panel.parquet"
RICH_PANEL_PATH = PROCESSED_ROOT / "dk1_panel_rich.parquet"
RICH_PANEL_ARCHIVE_PATH = PROCESSED_ROOT / "dk1_panel_rich_archive.parquet"
"""``_archive`` variant uses lag_hours=0 (RAW reanalysis) — only for the M3
F8b-archive leakage-quantification retrain. Every downstream consumer must
tag its outputs with ``-archive`` if it reads this file."""


def _quarter_index(start: datetime, end: datetime) -> pl.Series:
    import pandas as pd

    rng = pd.date_range(start=start, end=end, freq="15min", tz="UTC", inclusive="left")
    return pl.Series("timestamp_utc", rng.to_pydatetime().tolist()).cast(
        pl.Datetime("us", time_zone="UTC")
    )


def _to_15min_grid(
    base: pl.DataFrame, hourly: pl.DataFrame, value_cols: list[str]
) -> pl.DataFrame:
    """Left-join ``hourly`` onto the 15-min base grid and forward-fill values.

    ``hourly`` must have ``timestamp_utc`` and the listed value columns.
    """
    if hourly is None or hourly.height == 0:
        for c in value_cols:
            base = base.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
        return base
    joined = base.join(hourly.select(["timestamp_utc", *value_cols]), on="timestamp_utc", how="left")
    return joined.with_columns([pl.col(c).forward_fill() for c in value_cols])


def build_rich_panel(
    *,
    base_panel_path: Path | None = None,
    output_path: Path | None = None,
    weather_lag_hours: int = DEFAULT_WEATHER_LAG_HOURS,
    rebuild: bool = False,
) -> pl.DataFrame:
    """Build (or load cached) rich panel.

    Parameters
    ----------
    base_panel_path
        Canonical DK1 panel parquet (default ``data/processed/dk1_panel.parquet``).
    output_path
        Where to write the rich panel. Defaults to ``dk1_panel_rich.parquet`` or
        ``dk1_panel_rich_archive.parquet`` if ``weather_lag_hours == 0``.
    weather_lag_hours
        24 (default, honest) or 0 (archive, M3 only). See ``trinity.py``.
    rebuild
        Force recomputation even if the output already exists.
    """
    base_path = base_panel_path or BASE_PANEL_PATH
    if not base_path.exists():
        raise FileNotFoundError(
            f"Base panel not found at {base_path}. Run heimdall_data.loaders.load_dk1_panel first."
        )

    out_path = output_path or (
        RICH_PANEL_ARCHIVE_PATH if weather_lag_hours == 0 else RICH_PANEL_PATH
    )
    if out_path.exists() and not rebuild:
        return pl.read_parquet(out_path)

    base = pl.read_parquet(base_path).sort("timestamp_utc")
    if "timestamp_utc" not in base.columns:
        raise KeyError("base panel missing 'timestamp_utc'")

    start = base["timestamp_utc"].min()
    end = base["timestamp_utc"].max()
    # Convert polars datetime to native datetime for downstream helpers that
    # expect tz-aware Python datetimes.
    if start is None or end is None:
        raise ValueError("base panel is empty")

    # ----- Trinity wide features (hourly) -----
    try:
        trinity_wide = make_dk1_wide_features(
            start_utc=start, end_utc=end + _one_hour(), weather_lag_hours=weather_lag_hours
        )
    except FileNotFoundError:
        trinity_wide = None

    # ----- Markets (hourly grid, forward-filled from daily settles) -----
    markets_wide = load_markets(start, end + _one_hour())

    # ----- Merge onto 15-min base grid -----
    rich = base
    if trinity_wide is not None:
        # Drop dk1_da_price_eur_mwh (we already have da_price_dkk_mwh) to avoid
        # name collision and keep authoritative DKK price from existing pipeline.
        # Cross-zone EUR prices kept (they're new info).
        keep = [c for c in trinity_wide.columns if c != "dk1_da_price_eur_mwh"]
        trinity_wide = trinity_wide.select(keep)
        trinity_cols = [c for c in trinity_wide.columns if c != "timestamp_utc"]
        # umm cols are int, not float — keep them as-is, only forward-fill numerics.
        rich = _to_15min_grid(rich, trinity_wide, trinity_cols)

    market_cols = [c for c in markets_wide.columns if c != "timestamp_utc"]
    rich = _to_15min_grid(rich, markets_wide, market_cols)

    # ----- Calendar features (positional) -----
    rich = add_calendar_features(rich)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rich.write_parquet(out_path)
    return rich


def _one_hour():
    from datetime import timedelta

    return timedelta(hours=1)


__all__ = [
    "BASE_PANEL_PATH",
    "RICH_PANEL_ARCHIVE_PATH",
    "RICH_PANEL_PATH",
    "build_rich_panel",
]
