"""Calendar / cyclical features for the DK1 panel.

Per Plan v2 Track A.7 (notes/traders_trinity_audit.md). Produces deterministic,
strictly causal features from a UTC timestamp index:

  - cyclical: hour/dow/month/quarter as sin+cos pairs
  - holiday flags: DK + DE + SE (neighbouring zones; cross-border demand matters)
  - is_weekend, is_dst_transition_dk, days_to_next_dk_holiday

All features are POSITIONAL (depend only on the timestamp itself), so there is
no leakage risk regardless of train/val/test split.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
from holidays import country_holidays

DK_HOLIDAYS = country_holidays("DK", years=range(2015, 2031))
DE_HOLIDAYS = country_holidays("DE", years=range(2015, 2031))
SE_HOLIDAYS = country_holidays("SE", years=range(2015, 2031))


def _cyclical(values: pl.Expr, period: int, name: str) -> tuple[pl.Expr, pl.Expr]:
    """Return (sin, cos) of `2π·values/period` with names ``{name}_sin``, ``{name}_cos``."""
    angle = 2 * np.pi * values / period
    return angle.sin().alias(f"{name}_sin"), angle.cos().alias(f"{name}_cos")


def _next_holiday_distance_days(d: date, holiday_dates: set[date], max_days: int = 30) -> int:
    """Calendar-days until next DK holiday (capped at max_days)."""
    for offset in range(max_days + 1):
        if (d + timedelta(days=offset)) in holiday_dates:
            return offset
    return max_days


def add_calendar_features(panel: pl.DataFrame, ts_col: str = "timestamp_utc") -> pl.DataFrame:
    """Add cyclical + holiday + DST + days-to-holiday features in place (returns new frame).

    Local-time hour for DST detection uses Europe/Copenhagen (CET/CEST). DK and DE
    share that timezone; SE is the same offset year-round. For our purposes,
    we use Copenhagen consistently — the cross-border effect of slightly different
    DST transitions is in the noise compared to load patterns.
    """
    if ts_col not in panel.columns:
        raise KeyError(f"{ts_col!r} not in panel columns")

    ts = pl.col(ts_col)
    local = ts.dt.convert_time_zone("Europe/Copenhagen")

    hour_sin, hour_cos = _cyclical(local.dt.hour().cast(pl.Float64), 24, "hour")
    dow_sin, dow_cos = _cyclical(local.dt.weekday().cast(pl.Float64), 7, "dow")
    month_sin, month_cos = _cyclical(local.dt.month().cast(pl.Float64) - 1, 12, "month")
    quarter_sin, quarter_cos = _cyclical(local.dt.quarter().cast(pl.Float64) - 1, 4, "quarter")

    dk_set = set(DK_HOLIDAYS.keys())
    de_set = set(DE_HOLIDAYS.keys())
    se_set = set(SE_HOLIDAYS.keys())

    out = panel.with_columns(
        hour_sin,
        hour_cos,
        dow_sin,
        dow_cos,
        month_sin,
        month_cos,
        quarter_sin,
        quarter_cos,
        (local.dt.weekday() >= 6).alias("is_weekend"),
    )

    # Holiday flags + DST + days_to_next_holiday computed from the date column.
    local_date = out.select(
        pl.col(ts_col).dt.convert_time_zone("Europe/Copenhagen").dt.date().alias("_local_date")
    )["_local_date"]

    is_dk = pl.Series("is_dk_holiday", [d in dk_set for d in local_date], dtype=pl.Boolean)
    is_de = pl.Series("is_de_holiday", [d in de_set for d in local_date], dtype=pl.Boolean)
    is_se = pl.Series("is_se_holiday", [d in se_set for d in local_date], dtype=pl.Boolean)
    days_to = pl.Series(
        "days_to_next_dk_holiday",
        [_next_holiday_distance_days(d, dk_set) for d in local_date],
        dtype=pl.Int32,
    )

    # DST transition flag: True on the spring-forward and fall-back days.
    # A "transition day" in Europe/Copenhagen has 23 or 25 hours instead of 24.
    # We detect it by checking whether the date appears with multiple distinct
    # UTC-offsets in the panel — but a cheaper proxy: last Sunday of March or October.
    def _is_dst_day(d: date) -> bool:
        # Last Sunday of March (spring forward) or October (fall back)
        if d.month == 3:
            # Find last Sunday of March
            last_day = date(d.year, 3, 31)
            last_sun = last_day - timedelta(days=(last_day.weekday() - 6) % 7)
            return d == last_sun
        if d.month == 10:
            last_day = date(d.year, 10, 31)
            last_sun = last_day - timedelta(days=(last_day.weekday() - 6) % 7)
            return d == last_sun
        return False

    is_dst = pl.Series(
        "is_dst_transition", [_is_dst_day(d) for d in local_date], dtype=pl.Boolean
    )

    return out.with_columns(is_dk, is_de, is_se, is_dst, days_to)


__all__ = ["add_calendar_features"]
