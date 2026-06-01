"""Trinity vendored-features loader.

Reads the upstream traders-trinity parquets at ``data/external/trinity/*.parquet``
and exposes filter/join helpers consumable by ``load_dk1_panel``.

**Causality contract (CRITICAL)**: weather is ECMWF reanalysis (Open-Meteo archive
endpoint) which assimilates post-hoc observations. To avoid forward-information
leak we apply a **24-hour causal lag** at read time: the value returned for
``t`` is the Trinity value at ``t - 24h``. See ``data/external/trinity/PROVENANCE.md``
and ``notes/traders_trinity_audit.md`` for the full rationale.

The lag is **mandatory by default**. The ``WEATHER_LAG_HOURS = 0`` escape hatch
exists ONLY for the F8b-archive M3 leakage-quantification retrain.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

TRINITY_ROOT = Path(__file__).resolve().parents[4] / "data" / "external" / "trinity"

DEFAULT_WEATHER_LAG_HOURS = 24
"""Causal lag for Trinity weather (reanalysis). DO NOT change without explicit
M3 disclosure in the leaderboard / paper."""


def _resolve_root(root: Path | None) -> Path:
    r = root or TRINITY_ROOT
    if not r.exists():
        raise FileNotFoundError(
            f"Trinity vendored data not found at {r}. "
            "See data/external/trinity/PROVENANCE.md to re-import."
        )
    return r


def load_trinity_prices(
    *, zones: tuple[str, ...] = ("DK1",), root: Path | None = None
) -> pl.DataFrame:
    """DA prices, long form (utc_timestamp, zone, price_eur_mwh)."""
    r = _resolve_root(root)
    return (
        pl.read_parquet(r / "prices.parquet")
        .filter(pl.col("zone").is_in(list(zones)))
        .with_columns(
            pl.col("utc_timestamp").cast(pl.Datetime("us", time_zone="UTC")),
        )
        .sort(["zone", "utc_timestamp"])
    )


def load_trinity_load(
    *, zones: tuple[str, ...] = ("DK1",), root: Path | None = None
) -> pl.DataFrame:
    """Load actuals, long form."""
    r = _resolve_root(root)
    return (
        pl.read_parquet(r / "load.parquet")
        .filter(pl.col("zone").is_in(list(zones)))
        .with_columns(pl.col("utc_timestamp").cast(pl.Datetime("us", time_zone="UTC")))
        .sort(["zone", "utc_timestamp"])
    )


def load_trinity_generation(
    *, zones: tuple[str, ...] = ("DK1",), root: Path | None = None
) -> pl.DataFrame:
    """Generation actuals by fuel × zone (lagged-OK; never use t+h values as inputs)."""
    r = _resolve_root(root)
    return (
        pl.read_parquet(r / "generation.parquet")
        .filter(pl.col("zone").is_in(list(zones)))
        .with_columns(pl.col("utc_timestamp").cast(pl.Datetime("us", time_zone="UTC")))
        .sort(["zone", "utc_timestamp"])
    )


def load_trinity_flows(
    *,
    from_zones: tuple[str, ...] = ("DK1",),
    to_zones: tuple[str, ...] | None = None,
    root: Path | None = None,
) -> pl.DataFrame:
    """Cross-border flow actuals between specified zone pairs."""
    r = _resolve_root(root)
    df = pl.read_parquet(r / "flows.parquet").with_columns(
        pl.col("utc_timestamp").cast(pl.Datetime("us", time_zone="UTC"))
    )
    if from_zones:
        df = df.filter(pl.col("from_zone").is_in(list(from_zones)))
    if to_zones:
        df = df.filter(pl.col("to_zone").is_in(list(to_zones)))
    return df.sort(["from_zone", "to_zone", "utc_timestamp"])


def load_trinity_weather(
    *,
    zones: tuple[str, ...] = ("DK1",),
    lag_hours: int = DEFAULT_WEATHER_LAG_HOURS,
    root: Path | None = None,
) -> pl.DataFrame:
    """Weather features with **mandatory causal lag** applied.

    Parameters
    ----------
    zones
        Bidding zones to return.
    lag_hours
        Hours to shift the data forward so that ``out[t] := raw[t - lag_hours]``.
        Default 24h. Setting ``lag_hours = 0`` returns the raw archive value and
        is **only** allowed for the F8b-archive M3 leakage-quantification retrain
        — every downstream model that consumes a ``lag_hours = 0`` series must
        be tagged ``-archive`` in the leaderboard.

    Notes
    -----
    The lag is applied in *Trinity local time index space*: we add
    ``lag_hours`` to ``utc_timestamp`` for every row, then return. So a value
    that Trinity reported at hour `t` is exposed under key `t + lag_hours`.
    Equivalently: the row Heimdall looks up at hour `T` is the Trinity row at
    hour `T - lag_hours`.
    """
    if lag_hours < 0:
        raise ValueError(f"lag_hours must be non-negative; got {lag_hours}")
    r = _resolve_root(root)
    df = (
        pl.read_parquet(r / "weather.parquet")
        .filter(pl.col("zone").is_in(list(zones)))
        .with_columns(pl.col("utc_timestamp").cast(pl.Datetime("us", time_zone="UTC")))
    )
    if lag_hours > 0:
        df = df.with_columns(
            (pl.col("utc_timestamp") + pl.duration(hours=lag_hours)).alias("utc_timestamp")
        )
    return df.sort(["zone", "utc_timestamp"])


def load_trinity_umm(
    *, zones: tuple[str, ...] = ("DK1",), root: Path | None = None
) -> pl.DataFrame:
    """UMM outage features (causal by construction — published when event occurs)."""
    r = _resolve_root(root)
    f = r / "nordpool_umm" / "umm_zone_hourly_features.parquet"
    return (
        pl.read_parquet(f)
        .filter(pl.col("zone").is_in(list(zones)))
        .with_columns(pl.col("utc_timestamp").cast(pl.Datetime("us", time_zone="UTC")))
        .sort(["zone", "utc_timestamp"])
    )


def make_dk1_wide_features(
    start_utc,
    end_utc,
    *,
    weather_lag_hours: int = DEFAULT_WEATHER_LAG_HOURS,
    cross_zones_da: tuple[str, ...] = ("DE", "SE3", "NO2"),
    root: Path | None = None,
) -> pl.DataFrame:
    """Build a wide DK1-only feature frame from Trinity assets.

    Output schema (15-min grid is the responsibility of the caller via
    ``load_dk1_panel``; this returns the natural hourly grid):

      - timestamp_utc
      - dk1_da_price_eur_mwh
      - dk1_load_mw
      - wind_gen_mw           (Wind Offshore + Wind Onshore actuals, lagged-OK)
      - solar_gen_mw          (Solar actuals, lagged-OK)
      - hydro_reservoir_gen_mw (Hydro Water Reservoir actuals, lagged-OK)
      - fossil_gen_mw         (Fossil Gas + Fossil Hard coal, lagged-OK)
      - flow_dk1_de_mw, flow_dk1_se3_mw, flow_dk1_no2_mw  (lagged-OK actuals)
      - weather columns from Trinity (already 24h-lagged): wind_speed_100m,
        wind_direction_100m, temperature_2m, shortwave_radiation, cloud_cover,
        precipitation, surface_pressure
      - {de,se3,no2}_da_price_eur_mwh — cross-zone DA prices (known one day ahead)
      - umm_unavailable_capacity_mw, umm_active_event_count (causal)
    """
    # ----- prices: DK1 + cross-zone -----
    p = load_trinity_prices(zones=("DK1", "DK2", "DE", "SE3", "NO2", "FI"), root=root)
    dk1_da = (
        p.filter(pl.col("zone") == "DK1")
        .select(
            pl.col("utc_timestamp").alias("timestamp_utc"),
            pl.col("price_eur_mwh").alias("dk1_da_price_eur_mwh"),
        )
    )
    cross = (
        p.filter(pl.col("zone").is_in(list(cross_zones_da)))
        .with_columns(
            (pl.col("zone").str.to_lowercase() + "_da_price_eur_mwh").alias("col_name")
        )
        .pivot(values="price_eur_mwh", index="utc_timestamp", on="zone", aggregate_function="first")
        .rename({z: f"{z.lower()}_da_price_eur_mwh" for z in cross_zones_da})
        .rename({"utc_timestamp": "timestamp_utc"})
    )

    # ----- load -----
    ld = (
        load_trinity_load(zones=("DK1",), root=root)
        .select(
            pl.col("utc_timestamp").alias("timestamp_utc"),
            pl.col("load_mw").alias("dk1_load_mw"),
        )
    )

    # ----- generation aggregates -----
    g = load_trinity_generation(zones=("DK1",), root=root)
    g_cols = set(g.columns)

    def _coalesce_sum(cols: tuple[str, ...]) -> pl.Expr:
        present = [pl.col(c).fill_null(0.0) for c in cols if c in g_cols]
        if not present:
            return pl.lit(0.0)
        expr = present[0]
        for e in present[1:]:
            expr = expr + e
        return expr

    g_agg = g.select(
        pl.col("utc_timestamp").alias("timestamp_utc"),
        _coalesce_sum(("Wind Offshore", "Wind Onshore")).alias("wind_gen_mw"),
        _coalesce_sum(("Solar",)).alias("solar_gen_mw"),
        _coalesce_sum(("Hydro Water Reservoir",)).alias("hydro_reservoir_gen_mw"),
        _coalesce_sum(("Fossil Gas", "Fossil Hard coal", "Fossil Oil")).alias(
            "fossil_gen_mw"
        ),
    )

    # ----- flows -----
    fl = load_trinity_flows(from_zones=("DK1",), to_zones=("DE", "SE3", "NO2"), root=root)
    fl_w = (
        fl.with_columns(
            (pl.lit("flow_dk1_") + pl.col("to_zone").str.to_lowercase() + pl.lit("_mw")).alias(
                "col_name"
            )
        )
        .pivot(values="flow_mw", index="utc_timestamp", on="to_zone", aggregate_function="first")
        .rename({z: f"flow_dk1_{z.lower()}_mw" for z in ("DE", "SE3", "NO2")})
        .rename({"utc_timestamp": "timestamp_utc"})
    )

    # ----- weather (M2 24h lag applied inside load_trinity_weather) -----
    w = load_trinity_weather(zones=("DK1",), lag_hours=weather_lag_hours, root=root)
    w_dk1 = w.select(
        pl.col("utc_timestamp").alias("timestamp_utc"),
        pl.col("wind_speed_100m"),
        pl.col("wind_direction_100m"),
        pl.col("temperature_2m"),
        pl.col("shortwave_radiation"),
        pl.col("cloud_cover"),
        pl.col("precipitation"),
        pl.col("surface_pressure"),
    )

    # ----- UMM (causal) -----
    try:
        umm = load_trinity_umm(zones=("DK1",), root=root)
        umm_sel = umm.select(
            pl.col("utc_timestamp").alias("timestamp_utc"),
            pl.col("unavailable_capacity_mw").alias("umm_unavailable_capacity_mw"),
            pl.col("active_event_count").alias("umm_active_event_count"),
        )
    except Exception:
        umm_sel = None  # pre-2024-03 — column will be all null

    # ----- join all -----
    out = dk1_da
    for frame in (cross, ld, g_agg, fl_w, w_dk1):
        out = out.join(frame, on="timestamp_utc", how="left")
    if umm_sel is not None:
        out = out.join(umm_sel, on="timestamp_utc", how="left")

    # Optional time window filter
    if start_utc is not None or end_utc is not None:
        if start_utc is not None:
            out = out.filter(pl.col("timestamp_utc") >= start_utc)
        if end_utc is not None:
            out = out.filter(pl.col("timestamp_utc") < end_utc)

    return out.sort("timestamp_utc")


__all__ = [
    "DEFAULT_WEATHER_LAG_HOURS",
    "TRINITY_ROOT",
    "load_trinity_flows",
    "load_trinity_generation",
    "load_trinity_load",
    "load_trinity_prices",
    "load_trinity_umm",
    "load_trinity_weather",
    "make_dk1_wide_features",
]
