"""ENTSO-E Transparency Platform v2 REST wrapper.

Per docs/RESEARCH-PROPOSAL.md §5.1, ENTSO-E is the cross-EU baseline. We use the
``entsoe-py`` client (battle-tested, handles XML quirks) but expose a small
adapter so callers never touch ``EntsoeRawClient`` directly.

Endpoints we care about for DK1:

- ``A85`` Imbalance prices (per imbalance settlement period)
- ``A83`` Activated balancing energy (mFRR up/down volumes; business type A97)
- ``A65`` Total load — actual + day-ahead (ID ``A60``/``A61``)
- ``A44`` Day-ahead prices

The token is read from the ``ENTSOE_API_TOKEN`` environment variable; we never
log or echo it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from entsoe import EntsoePandasClient
from tenacity import retry, stop_after_attempt, wait_exponential

from heimdall_data.cache import CachedFrame, read_cached_frame, write_cached_frame

# Bidding zone code for DK1 in the ENTSO-E codelist.
DK1_BZN = "DK_1"
DK2_BZN = "DK_2"
ZONE_TO_BZN = {"DK1": DK1_BZN, "DK2": DK2_BZN}

GENERATION_TYPES: dict[str, tuple[str, ...]] = {
    "wind": ("Wind Offshore", "Wind Onshore"),
    "solar": ("Solar",),
    "hydro": ("Hydro Pumped Storage", "Hydro Run-of-river and poundage", "Hydro Water Reservoir"),
    "thermal": (
        "Biomass",
        "Fossil Brown coal/Lignite",
        "Fossil Coal-derived gas",
        "Fossil Gas",
        "Fossil Hard coal",
        "Fossil Oil",
        "Fossil Oil shale",
        "Fossil Peat",
        "Waste",
        "Other",
    ),
}


def _require_token(token: str | None) -> str:
    t = token or os.environ.get("ENTSOE_API_KEY") or os.environ.get("ENTSOE_API_TOKEN")
    if not t:
        raise RuntimeError(
            "ENTSO-E API token missing. Set ENTSOE_API_KEY or ENTSOE_API_TOKEN in the environment. "
            "Never hardcode API keys."
        )
    return t


@dataclass
class EntsoeClient:
    """Thin facade around ``entsoe.EntsoePandasClient``.

    Constructed lazily; tests can pass a stubbed ``client`` to short-circuit
    network access. We keep the surface deliberately narrow — only the four
    endpoints listed in the module docstring.
    """

    token: str | None = None
    bidding_zone: str = DK1_BZN
    _client: Any = None  # ``entsoe.EntsoePandasClient`` once constructed

    def __post_init__(self) -> None:
        self.token = _require_token(self.token)

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = EntsoePandasClient(api_key=self.token)
        return self._client

    # ----- public endpoints ---------------------------------------------

    def imbalance_prices(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """A85 — imbalance prices per settlement period (15 min on DK from 2025-03-04)."""
        c = self._ensure_client()
        df = c.query_imbalance_prices(self.bidding_zone, start=start, end=end)
        return _ensure_utc_index(df)

    def mfrr_activated_volumes(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """A83 — activated balancing energy by direction (mFRR up/down).

        ``entsoe-py`` returns wide-format with named columns per process+direction.
        Caller should select the mFRR rows.
        """
        c = self._ensure_client()
        df = c.query_activated_balancing_energy(
            self.bidding_zone, start=start, end=end, business_type="A97"  # mFRR
        )
        return _ensure_utc_index(df)

    def total_load(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        kind: str = "actual",
    ) -> pd.Series:
        """A65 — total load. ``kind`` ∈ {"actual", "forecast"} (forecast = day-ahead)."""
        c = self._ensure_client()
        if kind == "actual":
            return _ensure_utc_index(c.query_load(self.bidding_zone, start=start, end=end))
        if kind == "forecast":
            return _ensure_utc_index(
                c.query_load_forecast(self.bidding_zone, start=start, end=end)
            )
        raise ValueError(f"unknown kind={kind!r}")

    def day_ahead_prices(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """A44 — day-ahead clearing prices."""
        c = self._ensure_client()
        s = c.query_day_ahead_prices(self.bidding_zone, start=start, end=end)
        return _ensure_utc_index(s)

    def generation(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        generation_type: str = "all",
    ) -> pd.DataFrame:
        c = self._ensure_client()
        df = c.query_generation(self.bidding_zone, start=start, end=end, psr_type=None)
        df = _ensure_utc_index(_as_frame(df))
        if generation_type == "all":
            return df
        columns = GENERATION_TYPES.get(generation_type)
        if columns is None:
            raise ValueError(f"unknown generation_type={generation_type!r}")
        selected = [column for column in df.columns if str(column) in columns]
        if not selected:
            return pd.DataFrame(index=df.index)
        return df[selected]

    def crossborder_flows(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        counterparty: str,
    ) -> pd.Series:
        c = self._ensure_client()
        series = c.query_crossborder_flows(self.bidding_zone, counterparty, start=start, end=end)
        return _ensure_utc_index(series)


def normalize_price_series(
    series: pd.Series,
    *,
    zone: str,
    price_type: str,
) -> pd.DataFrame:
    frame = _series_to_frame(series, "price_eur_mwh")
    frame["zone"] = zone
    frame["price_type"] = price_type
    return frame[["timestamp_utc", "zone", "price_type", "price_eur_mwh"]]


def normalize_load_series(series: pd.Series | pd.DataFrame, *, zone: str, kind: str) -> pd.DataFrame:
    frame = _series_to_frame(_first_numeric_series(series), "load_mw")
    frame["zone"] = zone
    frame["kind"] = kind
    return frame[["timestamp_utc", "zone", "kind", "load_mw"]]


def normalize_generation_frame(
    frame: pd.DataFrame,
    *,
    zone: str,
    generation_type: str,
) -> pd.DataFrame:
    normalized = _as_frame(frame).copy()
    normalized.index.name = "timestamp_utc"
    tidy = normalized.reset_index().melt(
        id_vars=["timestamp_utc"],
        var_name="production_type",
        value_name="generation_mw",
    )
    tidy["timestamp_utc"] = pd.to_datetime(tidy["timestamp_utc"], utc=True)
    tidy["generation_mw"] = pd.to_numeric(tidy["generation_mw"], errors="coerce")
    tidy = tidy.dropna(subset=["generation_mw"])
    tidy["zone"] = zone
    tidy["generation_type"] = generation_type
    return tidy[
        ["timestamp_utc", "zone", "generation_type", "production_type", "generation_mw"]
    ].sort_values(["timestamp_utc", "production_type"]).reset_index(drop=True)


def normalize_flow_series(
    series: pd.Series,
    *,
    zone: str,
    counterparty: str,
) -> pd.DataFrame:
    frame = _series_to_frame(series, "flow_mw")
    frame["from_zone"] = zone
    frame["to_zone"] = counterparty
    return frame[["timestamp_utc", "from_zone", "to_zone", "flow_mw"]]


def normalize_activation_frame(frame: pd.DataFrame, *, zone: str) -> pd.DataFrame:
    raw = _as_frame(_ensure_utc_index(frame)).copy()
    if raw.empty:
        return pd.DataFrame(
            columns=["timestamp_utc", "zone", "activation_direction", "activated_volume_mwh"]
        )
    long = raw.reset_index().melt(
        id_vars=["timestamp_utc"],
        var_name="source_column",
        value_name="activated_volume_mwh",
    )
    long["activated_volume_mwh"] = pd.to_numeric(
        long["activated_volume_mwh"], errors="coerce"
    ).abs()
    long = long.dropna(subset=["activated_volume_mwh"])
    long = long[long["activated_volume_mwh"] > 0].copy()
    if long.empty:
        return pd.DataFrame(
            columns=["timestamp_utc", "zone", "activation_direction", "activated_volume_mwh"]
        )
    long["activation_direction"] = long["source_column"].map(_activation_direction_from_column)
    long = long.dropna(subset=["activation_direction"])
    grouped = (
        long.groupby(["timestamp_utc", "activation_direction"], as_index=False)[
            "activated_volume_mwh"
        ]
        .sum()
        .sort_values(["timestamp_utc", "activation_direction"])
    )
    grouped["zone"] = zone
    return grouped[
        ["timestamp_utc", "zone", "activation_direction", "activated_volume_mwh"]
    ].reset_index(drop=True)


def get_cached_entsoe_mfrr_activations(
    *,
    zone: str,
    start: datetime,
    end: datetime,
    refresh: bool = False,
    cache_dir: Path | None = None,
    client: EntsoeClient | None = None,
) -> CachedFrame:
    key = _cache_key("entsoe_mfrr_activations", zone, "a82", start, end)
    if not refresh:
        cached = read_cached_frame(key, cache_dir=cache_dir)
        if cached is not None:
            return cached
    active = client or EntsoeClient(bidding_zone=ZONE_TO_BZN[zone])
    frame = normalize_activation_frame(
        _retry_call(active.mfrr_activated_volumes, *_window(start, end)),
        zone=zone,
    )
    return _write_required(key, frame, "entsoe", cache_dir=cache_dir)


def get_cached_entsoe_prices(
    *,
    zone: str,
    price_type: str,
    start: datetime,
    end: datetime,
    refresh: bool = False,
    cache_dir: Path | None = None,
    client: EntsoeClient | None = None,
) -> CachedFrame:
    allowed = {"day_ahead", "imbalance", "mfrr_up", "mfrr_down"}
    if price_type not in allowed:
        raise ValueError(f"unknown price_type={price_type!r}")
    key = _cache_key("entsoe_prices", zone, price_type, start, end)
    if not refresh:
        cached = read_cached_frame(key, cache_dir=cache_dir)
        if cached is not None:
            return cached
    active = client or EntsoeClient(bidding_zone=ZONE_TO_BZN[zone])
    start_ts, end_ts = _window(start, end)
    if price_type == "day_ahead":
        frame = normalize_price_series(
            _retry_call(active.day_ahead_prices, start_ts, end_ts),
            zone=zone,
            price_type=price_type,
        )
    else:
        raw = _as_frame(_retry_call(active.imbalance_prices, start_ts, end_ts))
        frame = _normalize_imbalance_prices(raw, zone=zone, price_type=price_type)
    return _write_required(key, frame, "entsoe", cache_dir=cache_dir)


def get_cached_entsoe_loads(
    *,
    zone: str,
    kind: str,
    start: datetime,
    end: datetime,
    refresh: bool = False,
    cache_dir: Path | None = None,
    client: EntsoeClient | None = None,
) -> CachedFrame:
    if kind not in {"actual", "forecast"}:
        raise ValueError(f"unknown load kind={kind!r}")
    key = _cache_key("entsoe_loads", zone, kind, start, end)
    if not refresh:
        cached = read_cached_frame(key, cache_dir=cache_dir)
        if cached is not None:
            return cached
    active = client or EntsoeClient(bidding_zone=ZONE_TO_BZN[zone])
    frame = normalize_load_series(
        _retry_call(active.total_load, *_window(start, end), kind=kind),
        zone=zone,
        kind=kind,
    )
    return _write_required(key, frame, "entsoe", cache_dir=cache_dir)


def get_cached_entsoe_generation(
    *,
    zone: str,
    generation_type: str,
    start: datetime,
    end: datetime,
    refresh: bool = False,
    cache_dir: Path | None = None,
    client: EntsoeClient | None = None,
) -> CachedFrame:
    if generation_type != "all" and generation_type not in GENERATION_TYPES:
        raise ValueError(f"unknown generation_type={generation_type!r}")
    key = _cache_key("entsoe_generation", zone, generation_type, start, end)
    if not refresh:
        cached = read_cached_frame(key, cache_dir=cache_dir)
        if cached is not None:
            return cached
    active = client or EntsoeClient(bidding_zone=ZONE_TO_BZN[zone])
    frame = normalize_generation_frame(
        _retry_call(active.generation, *_window(start, end), generation_type),
        zone=zone,
        generation_type=generation_type,
    )
    return _write_required(key, frame, "entsoe", cache_dir=cache_dir)


def get_cached_entsoe_flows(
    *,
    zone: str,
    counterparty: str,
    start: datetime,
    end: datetime,
    refresh: bool = False,
    cache_dir: Path | None = None,
    client: EntsoeClient | None = None,
) -> CachedFrame:
    key = _cache_key("entsoe_flows", zone, counterparty, start, end)
    if not refresh:
        cached = read_cached_frame(key, cache_dir=cache_dir)
        if cached is not None:
            return cached
    active = client or EntsoeClient(bidding_zone=ZONE_TO_BZN[zone])
    frame = normalize_flow_series(
        _retry_call(active.crossborder_flows, *_window(start, end), counterparty=counterparty),
        zone=zone,
        counterparty=counterparty,
    )
    return _write_required(key, frame, "entsoe", cache_dir=cache_dir)


def _ensure_utc_index(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Force the index to UTC. ENTSO-E returns localised CET/CEST timestamps."""
    idx = obj.index
    if getattr(idx, "tz", None) is None:
        obj = obj.tz_localize("UTC")
    else:
        obj = obj.tz_convert("UTC")
    obj.index.name = "timestamp_utc"
    return obj


def _as_frame(obj: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        return obj.to_frame(name=obj.name or "value")
    return obj


def _series_to_frame(series: pd.Series, value_name: str) -> pd.DataFrame:
    clean = _ensure_utc_index(series).rename(value_name).reset_index()
    clean["timestamp_utc"] = pd.to_datetime(clean["timestamp_utc"], utc=True)
    clean[value_name] = pd.to_numeric(clean[value_name], errors="coerce")
    return clean.dropna(subset=[value_name]).sort_values("timestamp_utc").reset_index(drop=True)


def _first_numeric_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj
    numeric = obj.apply(pd.to_numeric, errors="coerce")
    for column in numeric.columns:
        if numeric[column].notna().any():
            return numeric[column]
    return pd.Series(dtype=float, index=obj.index)


def _normalize_imbalance_prices(raw: pd.DataFrame, *, zone: str, price_type: str) -> pd.DataFrame:
    frame = _as_frame(_ensure_utc_index(raw)).copy()
    if frame.empty:
        return pd.DataFrame(columns=["timestamp_utc", "zone", "price_type", "price_eur_mwh"])
    lowered = {str(column).lower(): column for column in frame.columns}
    candidates = {
        "imbalance": ["imbalance_price_eur_mwh", "imbalance price", "imbalance"],
        "mfrr_up": ["mfrr_up", "up", "marginal price up", "positive"],
        "mfrr_down": ["mfrr_down", "down", "marginal price down", "negative"],
    }[price_type]
    selected = None
    for needle in candidates:
        for lower, column in lowered.items():
            if needle in lower:
                selected = column
                break
        if selected is not None:
            break
    if selected is None:
        selected = frame.columns[0]
    return normalize_price_series(frame[selected], zone=zone, price_type=price_type)


def _activation_direction_from_column(column: object) -> str | None:
    lowered = str(column).lower()
    up_tokens = ("up", "positive", "upward")
    down_tokens = ("down", "negative", "downward")
    if any(token in lowered for token in up_tokens):
        return "up"
    if any(token in lowered for token in down_tokens):
        return "down"
    return None


def _window(start: datetime, end: datetime) -> tuple[pd.Timestamp, pd.Timestamp]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("ENTSO-E windows must be timezone-aware")
    return pd.Timestamp(start.astimezone(UTC)), pd.Timestamp(end.astimezone(UTC))


def _cache_key(prefix: str, zone: str, kind: str, start: datetime, end: datetime) -> str:
    return (
        f"{prefix}_{zone}_{kind}_{start.astimezone(UTC).strftime('%Y%m%dT%H%M%S')}_"
        f"{end.astimezone(UTC).strftime('%Y%m%dT%H%M%S')}"
    )


def _write_required(key: str, frame: pd.DataFrame, source: str, *, cache_dir: Path | None) -> CachedFrame:
    if frame.empty:
        raise RuntimeError(f"{source} returned no rows for {key}")
    return write_cached_frame(key, frame, source=source, cache_dir=cache_dir)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _retry_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    return func(*args, **kwargs)


__all__ = [
    "DK1_BZN",
    "DK2_BZN",
    "GENERATION_TYPES",
    "EntsoeClient",
    "get_cached_entsoe_flows",
    "get_cached_entsoe_generation",
    "get_cached_entsoe_loads",
    "get_cached_entsoe_mfrr_activations",
    "get_cached_entsoe_prices",
    "normalize_activation_frame",
    "normalize_flow_series",
    "normalize_generation_frame",
    "normalize_load_series",
    "normalize_price_series",
]
