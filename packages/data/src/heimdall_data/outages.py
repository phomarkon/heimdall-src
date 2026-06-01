from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

UMM_MESSAGES_URL = "https://ummapi.nordpoolgroup.com/messages"
DK_NEIGHBOR_ZONES = {"DK1", "DK2", "NO2", "SE3", "SE4", "DE"}


class OutageDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class OutageEvent:
    id: str
    title: str
    zones: list[str]
    published_at_utc: str
    time_start_utc: str | None
    time_end_utc: str | None
    max_unavailable_capacity_mw: float


@dataclass
class UMMClient:
    session: requests.Session | None = None
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    def fetch_messages(
        self,
        *,
        publication_start: datetime,
        limit: int = 500,
        max_pages: int = 100,
    ) -> list[dict[str, Any]]:
        if publication_start.tzinfo is None:
            raise ValueError("publication_start must be timezone-aware")
        rows: list[dict[str, Any]] = []
        skip = 0
        pages = 0
        while True:
            if pages >= max_pages:
                raise OutageDataError(f"UMM pagination exceeded max_pages={max_pages}")
            batch = self._get(
                {
                    "filter.limit": limit,
                    "filter.skip": skip,
                    "filter.publicationStartDate": publication_start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                }
            )
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < limit:
                break
            skip += limit
            pages += 1
        return rows

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _get(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        assert self.session is not None
        response = self.session.get(UMM_MESSAGES_URL, params=params, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            raise OutageDataError(f"GET Nord Pool UMM -> {response.status_code}: {response.text[:200]}")
        payload = response.json()
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("messages", "items", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        raise OutageDataError("Nord Pool UMM response did not contain a message list")


def build_outage_events(
    messages: list[dict[str, Any]],
    *,
    zones: set[str] = DK_NEIGHBOR_ZONES,
    massive_threshold_mw: float = 500.0,
    min_capacity_mw: float = 150.0,
    title_similarity: float = 0.8,
    cap: int = 100,
) -> list[OutageEvent]:
    candidates: list[tuple[str, int, OutageEvent]] = []
    for message in messages:
        event = normalize_outage_message(message)
        if event is None:
            continue
        if event.max_unavailable_capacity_mw <= min_capacity_mw:
            continue
        if not (set(event.zones) & zones) and event.max_unavailable_capacity_mw < massive_threshold_mw:
            continue
        candidates.append((_message_id(message), _message_version(message), event))
    by_id: dict[str, tuple[int, OutageEvent]] = {}
    for message_id, version, event in candidates:
        previous = by_id.get(message_id)
        if previous is None or version >= previous[0]:
            by_id[message_id] = (version, event)
    deduped: list[OutageEvent] = []
    for _, event in by_id.values():
        if any(_similar(event.title, existing.title) >= title_similarity for existing in deduped):
            continue
        deduped.append(event)
    deduped.sort(
        key=lambda item: (
            item.published_at_utc or "",
            item.max_unavailable_capacity_mw,
        ),
        reverse=True,
    )
    return deduped[:cap]


def normalize_outage_message(message: dict[str, Any]) -> OutageEvent | None:
    capacity = _max_capacity(message)
    if capacity is None:
        return None
    title = str(_first(message, ["title", "eventTitle", "messageTitle", "subject"]) or "").strip()
    if not title:
        title = _synthetic_title(message)
    start, end = _event_window(message)
    return OutageEvent(
        id=_message_id(message),
        title=title,
        zones=sorted(_zones(message)),
        published_at_utc=_iso_or_empty(_first(message, ["publicationDate", "publishedAt", "createdDate", "publicationStartDate"])),
        time_start_utc=start,
        time_end_utc=end,
        max_unavailable_capacity_mw=float(capacity),
    )


def write_outages(path: Path, events: list[OutageEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([event.__dict__ for event in events], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _max_capacity(value: Any) -> float | None:
    found: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).lower() == "unavailablecapacity":
                    numeric = _to_float(child)
                    if numeric is not None:
                        found.append(numeric)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return max(found) if found else None


def _zones(value: Any) -> set[str]:
    out: set[str] = set()

    def add(raw: Any) -> None:
        text = str(raw or "").upper()
        for zone in DK_NEIGHBOR_ZONES:
            if re.search(rf"\b{re.escape(zone)}\b", text):
                out.add(zone)
        if text in {"DK_1", "DK WEST", "DK-WEST"}:
            out.add("DK1")
        if text in {"DK_2", "DK EAST", "DK-EAST"}:
            out.add("DK2")

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {"areaName", "inAreaName", "outAreaName", "name", "code"}:
                    add(child)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return out


def _event_window(message: dict[str, Any]) -> tuple[str | None, str | None]:
    starts: list[str] = []
    ends: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {"eventStart", "startTime", "unavailabilityStart", "timeStart"}:
                    parsed = _iso_or_none(child)
                    if parsed is not None:
                        starts.append(parsed)
                if key in {"eventStop", "endTime", "unavailabilityEnd", "timeEnd"}:
                    parsed = _iso_or_none(child)
                    if parsed is not None:
                        ends.append(parsed)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(message)
    return (min(starts) if starts else None, max(ends) if ends else None)


def _synthetic_title(message: dict[str, Any]) -> str:
    reason = str(_first(message, ["unavailabilityReason", "remarks"]) or "Outage").strip() or "Outage"
    names: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            name = item.get("productionUnitName") or item.get("name")
            if name and str(name) not in names:
                names.append(str(name))
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    for key in ("generationUnits", "productionUnits", "consumptionUnits", "transmissionUnits", "units"):
        visit(message.get(key, []))
    subject = ", ".join(names[:2]) if names else _message_id(message)
    return f"{reason}: {subject}"


def _first(message: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in message and message[key] not in (None, ""):
            return message[key]
    return None


def _message_id(message: dict[str, Any]) -> str:
    return str(_first(message, ["messageId", "id", "ummId"]) or "unknown")


def _message_version(message: dict[str, Any]) -> int:
    value = _first(message, ["version", "messageVersion"])
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_or_empty(value: Any) -> str:
    return _iso_or_none(value) or ""


def _iso_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _similar(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left.lower(), right.lower()).ratio()


__all__ = [
    "DK_NEIGHBOR_ZONES",
    "UMM_MESSAGES_URL",
    "OutageDataError",
    "OutageEvent",
    "UMMClient",
    "build_outage_events",
    "normalize_outage_message",
    "write_outages",
]
