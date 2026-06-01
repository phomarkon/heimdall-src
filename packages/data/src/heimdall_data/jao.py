from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

JAO_BASE_URL = "https://publicationtool.jao.eu"
JAO_DEFAULT_ENDPOINTS = ("nordic:fbDomainShadowPrice", "core:activeFbConstraints")
JAO_REQUIRED_COLUMNS = [
    "timestamp_utc",
    "publication_time_utc",
    "zone",
    "cnec_id",
    "constraint_name",
    "ram_mw",
    "shadow_price_eur_mw",
    "flow_mw",
    "direction",
    "source_url",
]


class JAODataError(RuntimeError):
    pass


@dataclass(frozen=True)
class JAOConstraintRecord:
    timestamp_utc: str
    publication_time_utc: str
    zone: str
    cnec_id: str
    constraint_name: str
    ram_mw: float | None
    shadow_price_eur_mw: float | None
    flow_mw: float | None
    direction: str | None
    source_url: str


@dataclass
class JAOClient:
    api_key: str | None = None
    base_url: str | None = None
    session: requests.Session | None = None
    timeout_seconds: float = 60.0
    live_fetch_enabled: bool | None = None
    endpoints: tuple[str, ...] = JAO_DEFAULT_ENDPOINTS
    page_size: int = 5000

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
        if self.api_key is None:
            self.api_key = os.environ.get("JAO_API_KEY", "").strip() or None
        if self.base_url is None:
            self.base_url = os.environ.get("JAO_BASE_URL", "").strip() or JAO_BASE_URL
        if self.live_fetch_enabled is None:
            self.live_fetch_enabled = os.environ.get("JAO_ENABLE_LIVE_FETCH", "false").strip().lower() == "true"

    def fetch_constraints(self, *, start: datetime, end: datetime, zone: str) -> list[dict[str, Any]]:
        if not self.live_fetch_enabled:
            return []
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start/end must be timezone-aware")
        assert self.session is not None
        rows: list[dict[str, Any]] = []
        for endpoint in self.endpoints:
            rows.extend(self._fetch_endpoint(endpoint=endpoint, start=start, end=end, zone=zone))
        return rows

    def _fetch_endpoint(self, *, endpoint: str, start: datetime, end: datetime, zone: str) -> list[dict[str, Any]]:
        assert self.session is not None
        ccr, api_name = _parse_endpoint(endpoint)
        url = f"{self.base_url.rstrip('/')}/{ccr}/api/data/{api_name}"
        headers = {"AUTH_API_KEY": self.api_key} if self.api_key else {}
        rows: list[dict[str, Any]] = []
        skip = 0
        page_size = max(1, int(self.page_size))
        while True:
            response = self.session.get(
                url,
                params={
                    "FromUtc": _iso_z(start),
                    "ToUtc": _iso_z(end),
                    "Skip": skip,
                    "Take": page_size,
                },
                headers=headers,
                timeout=self.timeout_seconds,
            )
            if response.status_code >= 400:
                raise JAODataError(f"GET {url} -> {response.status_code}: {response.text[:200]}")
            payload = response.json()
            page = _extract_rows(payload)
            for row in page:
                row = dict(row)
                row.setdefault("_jao_endpoint", endpoint)
                row.setdefault("_jao_zone_query", zone)
                rows.append(row)
            total = _payload_total(payload)
            if not page or len(page) < page_size or total is None or skip + len(page) >= total:
                break
            skip += len(page)
        return rows


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "items", "results", "constraints"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise JAODataError("JAO response did not contain a constraint list")


def _payload_total(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in ("totalRowsWithFilter", "totalRows"):
        try:
            return int(payload[key])
        except (KeyError, TypeError, ValueError):
            pass
    return None


def normalize_constraint_rows(rows: list[dict[str, Any]], *, zone: str, source_url: str = JAO_BASE_URL) -> pd.DataFrame:
    records = [record.__dict__ for record in (_normalize_row(row, zone=zone, source_url=source_url) for row in rows) if record is not None]
    if not records:
        return empty_constraints_frame()
    frame = pd.DataFrame(records)
    return _canonical_frame(frame)


def empty_constraints_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in JAO_REQUIRED_COLUMNS})


def write_constraints(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _canonical_frame(frame).to_parquet(path, index=False)


def read_constraints(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_constraints_frame()
    return _canonical_frame(pd.read_parquet(path))


def get_cached_jao_constraints(
    *,
    start: datetime,
    end: datetime,
    zone: str,
    cache_dir: Path | None = None,
    refresh: bool = False,
    client: JAOClient | None = None,
) -> pd.DataFrame:
    cache_root = cache_dir or Path(os.environ.get("JAO_CACHE_DIR", "data/raw/jao"))
    cache_path = cache_root / f"jao_constraints_{zone}_{_window_hash(start, end)}.parquet"
    if cache_path.exists() and not refresh:
        return read_constraints(cache_path)
    client = client or JAOClient()
    rows = client.fetch_constraints(start=start, end=end, zone=zone)
    frame = normalize_constraint_rows(rows, zone=zone, source_url=client.base_url or JAO_BASE_URL)
    write_constraints(cache_path, frame)
    return frame


def unavailable_constraints_result(*, zone: str, reason: str = "jao_live_fetch_disabled") -> dict[str, Any]:
    return {
        "ok": True,
        "kind": "grid_constraints",
        "authority": "jao_optional",
        "zone": zone,
        "row_count": 0,
        "pressure_label": "unavailable",
        "signals": {"reason": reason},
    }


def _normalize_row(row: dict[str, Any], *, zone: str, source_url: str) -> JAOConstraintRecord | None:
    timestamp = _iso_or_none(_first(row, ["timestamp_utc", "timestamp", "deliveryTimestamp", "businessTimestamp", "timeIntervalStart", "dateTimeUtc"]))
    publication = _iso_or_none(_first(row, ["publication_time_utc", "publicationTime", "publicationTimestamp", "createdAt", "updatedAt", "lastModifiedOn"]))
    if timestamp is None:
        return None
    cnec_id = _first(row, ["cnec_id", "cnecId", "cnecEic", "cneEic", "constraintId", "id"]) or "unknown"
    selected_zone = _zone_from_row(row, fallback=zone)
    return JAOConstraintRecord(
        timestamp_utc=timestamp,
        publication_time_utc=publication or timestamp,
        zone=selected_zone,
        cnec_id=str(cnec_id),
        constraint_name=str(_first(row, ["constraint_name", "constraintName", "cnecName", "cneName", "name"]) or "JAO constraint"),
        ram_mw=_to_float(_first(row, ["ram_mw", "ram", "remainingAvailableMargin", "RAM"])),
        shadow_price_eur_mw=_to_float(_first(row, ["shadow_price_eur_mw", "shadowPrice", "shadowPriceEurMw", "marginalPrice"])),
        flow_mw=_to_float(_first(row, ["flow_mw", "flow", "flowFb", "fref", "frefInit", "allocatedFlow", "commercialFlow"])),
        direction=_direction(row),
        source_url=str(row.get("_jao_endpoint") or source_url),
    )


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for column in JAO_REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    out = frame[JAO_REQUIRED_COLUMNS].copy()
    for column in ("timestamp_utc", "publication_time_utc"):
        out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    for column in ("ram_mw", "shadow_price_eur_mw", "flow_mw"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values(["timestamp_utc", "publication_time_utc", "cnec_id"]).reset_index(drop=True)


def _first(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC).isoformat().replace("+00:00", "Z")


def _direction(row: dict[str, Any]) -> str | None:
    raw = _first(row, ["direction", "borderDirection", "hubFromTo", "fromTo"])
    if raw:
        return str(raw)
    left = _first(row, ["from", "fromArea", "fromZone", "biddingZoneFrom", "hubFrom"])
    right = _first(row, ["to", "toArea", "toZone", "biddingZoneTo", "hubTo"])
    if left or right:
        return f"{left or '?'}>{right or '?'}"
    return None


def _zone_from_row(row: dict[str, Any], *, fallback: str) -> str:
    explicit = _first(row, ["zone", "biddingZone", "area", "hub"])
    if explicit:
        return str(explicit)
    left = _first(row, ["biddingZoneFrom", "hubFrom", "fromZone", "fromArea", "from"])
    right = _first(row, ["biddingZoneTo", "hubTo", "toZone", "toArea", "to"])
    fallback = str(fallback)
    if fallback and fallback not in {"ALL", "*"} and fallback in {str(left), str(right)}:
        return fallback
    if left or right:
        return f"{left or '?'}>{right or '?'}"
    return fallback


def _parse_endpoint(endpoint: str) -> tuple[str, str]:
    if ":" not in endpoint:
        raise ValueError(f"JAO endpoint must be formatted as '<ccr>:<api_name>', got {endpoint!r}")
    ccr, api_name = endpoint.split(":", 1)
    if not ccr or not api_name:
        raise ValueError(f"JAO endpoint must be formatted as '<ccr>:<api_name>', got {endpoint!r}")
    return ccr, api_name


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _window_hash(start: datetime, end: datetime) -> str:
    payload = json.dumps([start.astimezone(UTC).isoformat(), end.astimezone(UTC).isoformat()])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


__all__ = [
    "JAOClient",
    "JAOConstraintRecord",
    "JAODataError",
    "empty_constraints_frame",
    "get_cached_jao_constraints",
    "normalize_constraint_rows",
    "read_constraints",
    "unavailable_constraints_result",
    "write_constraints",
]
