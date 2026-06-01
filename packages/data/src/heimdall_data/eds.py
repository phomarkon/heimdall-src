from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from heimdall_data.cache import CachedFrame, read_cached_frame, write_cached_frame

EDS_DATASET_URL = "https://api.energidataservice.dk/dataset"
DEFAULT_PAGE_SIZE = 5000
DEFAULT_DELAY_SECONDS = 1.25


class EDSDataError(RuntimeError):
    pass


@dataclass
class EDSRequestLog:
    dataset: str
    params: dict[str, Any]
    status_code: int | None
    rows: int
    duration_seconds: float
    cache_hit: bool = False
    error: str | None = None


@dataclass
class EDSClient:
    session: requests.Session | None = None
    timeout_seconds: float = 60.0
    page_size: int = DEFAULT_PAGE_SIZE
    delay_seconds: float = DEFAULT_DELAY_SECONDS
    request_log: list[EDSRequestLog] = field(default_factory=list)
    _last_request_at: float = 0.0

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")

    def fetch_dataset(
        self,
        dataset: str,
        *,
        start: datetime,
        end: datetime,
        filters: dict[str, list[str] | str] | None = None,
        sort: str = "TimeUTC asc",
        max_pages: int = 10_000,
    ) -> pd.DataFrame:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("EDS windows must be timezone-aware")
        if start >= end:
            raise ValueError("EDS window start must be before end")
        rows: list[dict[str, Any]] = []
        offset = 0
        pages = 0
        while True:
            if pages >= max_pages:
                raise EDSDataError(f"EDS pagination exceeded max_pages={max_pages} for {dataset}")
            params: dict[str, Any] = {
                "start": _eds_time(start),
                "end": _eds_time(end),
                "sort": sort,
                "limit": self.page_size,
                "offset": offset,
            }
            if filters:
                params["filter"] = _filter_json(filters)
            payload = self._get(dataset, params)
            batch = payload.get("records", [])
            if not isinstance(batch, list):
                raise EDSDataError(f"EDS {dataset} payload records is not a list")
            rows.extend(batch)
            pages += 1
            if len(batch) < self.page_size:
                break
            offset += self.page_size
        return pd.DataFrame(rows)

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _get(self, dataset: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self.session is not None
        self._throttle()
        started = time.monotonic()
        status_code: int | None = None
        try:
            response = self.session.get(
                f"{EDS_DATASET_URL}/{dataset}",
                params=params,
                timeout=self.timeout_seconds,
            )
            status_code = response.status_code
            if response.status_code == 429:
                retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
                time.sleep(max(retry_after, self.delay_seconds * 2.0))
                raise requests.Timeout("EDS rate limited")
            if response.status_code >= 400:
                raise EDSDataError(f"GET EDS {dataset} -> {response.status_code}: {response.text[:200]}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise EDSDataError(f"EDS {dataset} response was not a JSON object")
            rows = payload.get("records", [])
            self.request_log.append(
                EDSRequestLog(
                    dataset=dataset,
                    params=_safe_params(params),
                    status_code=status_code,
                    rows=len(rows) if isinstance(rows, list) else 0,
                    duration_seconds=round(time.monotonic() - started, 4),
                )
            )
            return payload
        except Exception as exc:
            self.request_log.append(
                EDSRequestLog(
                    dataset=dataset,
                    params=_safe_params(params),
                    status_code=status_code,
                    rows=0,
                    duration_seconds=round(time.monotonic() - started, 4),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self.delay_seconds - (now - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()


def get_cached_eds_dataset(
    dataset: str,
    *,
    start: datetime,
    end: datetime,
    filters: dict[str, list[str] | str] | None = None,
    sort: str = "TimeUTC asc",
    allow_empty: bool = False,
    refresh: bool = False,
    cache_dir: Path | None = None,
    client: EDSClient | None = None,
) -> CachedFrame:
    key = _cache_key("eds", dataset, f"{_filter_slug(filters)}_{sort.split(maxsplit=1)[0]}", start, end)
    if not refresh:
        cached = read_cached_frame(key, cache_dir=cache_dir)
        if cached is not None:
            if client is not None:
                client.request_log.append(
                    EDSRequestLog(dataset=dataset, params={"cache_key": key}, status_code=None, rows=len(cached.frame), duration_seconds=0.0, cache_hit=True)
                )
            return cached
    active = client or EDSClient()
    frame = active.fetch_dataset(dataset, start=start, end=end, filters=filters, sort=sort)
    if frame.empty and not allow_empty:
        raise EDSDataError(f"EDS {dataset} returned no rows for {key}")
    return write_cached_frame(
        key,
        frame,
        source="energidataservice",
        cache_dir=cache_dir,
        metadata={
            "dataset": dataset,
            "filters": filters or {},
            "window_start_utc": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "window_end_utc": end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        },
    )


def normalize_eds_table(frame: pd.DataFrame, *, dataset: str, zone_column: str = "PriceArea") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    time_column = _first_existing(frame, ["TimeUTC", "HourUTC", "Minutes5UTC"])
    if time_column is None:
        raise EDSDataError(f"EDS {dataset} missing UTC timestamp column")
    out = frame.copy()
    out["timestamp_utc"] = pd.to_datetime(out[time_column], utc=True, errors="coerce")
    if out["timestamp_utc"].isna().any():
        raise EDSDataError(f"EDS {dataset} contains invalid UTC timestamps")
    if zone_column in out.columns:
        out["zone"] = out[zone_column].astype(str)
    out["dataset"] = dataset
    for column in out.columns:
        if column in {"timestamp_utc", "zone", "dataset"}:
            continue
        converted = pd.to_numeric(out[column], errors="coerce")
        if converted.notna().any():
            out[column] = converted
    front = [column for column in ["timestamp_utc", "zone", "dataset"] if column in out.columns]
    rest = [column for column in out.columns if column not in front]
    return out[front + rest].sort_values(front or ["timestamp_utc"]).reset_index(drop=True)


def _first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _filter_json(filters: dict[str, list[str] | str]) -> str:
    payload: dict[str, Any] = {}
    for key, value in filters.items():
        payload[key] = value if isinstance(value, list) else [value]
    return json.dumps(payload, separators=(",", ":"))


def _filter_slug(filters: dict[str, list[str] | str] | None) -> str:
    if not filters:
        return "all"
    parts = []
    for key, value in sorted(filters.items()):
        values = value if isinstance(value, list) else [value]
        parts.append(f"{key}-{'-'.join(str(v) for v in values)}")
    return "_".join(parts).replace("/", "-")


def _eds_time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M")


def _cache_key(prefix: str, dataset: str, kind: str, start: datetime, end: datetime) -> str:
    return (
        f"{prefix}_{dataset}_{kind}_{start.astimezone(UTC).strftime('%Y%m%dT%H%M%S')}_"
        f"{end.astimezone(UTC).strftime('%Y%m%dT%H%M%S')}"
    )


def _retry_after_seconds(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _safe_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if "key" not in key.lower() and "token" not in key.lower()}


__all__ = [
    "DEFAULT_DELAY_SECONDS",
    "DEFAULT_PAGE_SIZE",
    "EDSClient",
    "EDSDataError",
    "EDSRequestLog",
    "get_cached_eds_dataset",
    "normalize_eds_table",
]
