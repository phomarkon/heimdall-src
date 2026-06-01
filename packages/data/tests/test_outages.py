from __future__ import annotations

from datetime import UTC, datetime

from heimdall_data.outages import UMMClient, build_outage_events


def test_umm_client_paginates_until_empty() -> None:
    class _Response:
        status_code = 200
        text = "{}"

        def __init__(self, rows: list[dict]) -> None:
            self._rows = rows

        def json(self) -> dict:
            return {"messages": self._rows}

    calls: list[dict] = []

    class _Session:
        def get(self, _url: str, *, params: dict, timeout: float) -> _Response:
            calls.append(dict(params))
            if params["filter.skip"] == 0:
                return _Response([{"messageId": "a"}] * 2)
            return _Response([])

    rows = UMMClient(session=_Session()).fetch_messages(  # type: ignore[arg-type]
        publication_start=datetime(2026, 4, 1, tzinfo=UTC),
        limit=2,
    )
    assert len(rows) == 2
    assert calls[0]["filter.limit"] == 2
    assert calls[1]["filter.skip"] == 2


def test_outage_normalization_filters_deduplicates_and_caps() -> None:
    messages = [
        {
            "messageId": "m1",
            "version": 1,
            "title": "DK1 generator unavailable",
            "publicationDate": "2026-04-03T12:00:00Z",
            "eventStart": "2026-04-03T13:00:00Z",
            "eventStop": "2026-04-04T13:00:00Z",
            "areas": [{"name": "DK1"}],
            "units": [{"areaName": "DK1", "timePeriods": [{"unavailableCapacity": 200}]}],
        },
        {
            "messageId": "m1",
            "version": 2,
            "title": "DK1 generator unavailable",
            "publicationDate": "2026-04-03T13:00:00Z",
            "areas": [{"name": "DK1"}],
            "units": [{"areaName": "DK1", "timePeriods": [{"unavailableCapacity": 260}]}],
        },
        {
            "messageId": "m2",
            "title": "Tiny outage",
            "publicationDate": "2026-04-02T12:00:00Z",
            "areas": [{"name": "DK2"}],
            "units": [{"areaName": "DK2", "unavailableCapacity": 100}],
        },
        {
            "messageId": "m3",
            "title": "Massive foreign outage",
            "publicationDate": "2026-04-01T12:00:00Z",
            "areas": [{"name": "FR"}],
            "units": [{"areaName": "FR", "unavailableCapacity": 600}],
        },
    ]
    events = build_outage_events(messages)
    assert [event.id for event in events] == ["m1", "m3"]
    assert events[0].max_unavailable_capacity_mw == 260
    assert events[0].zones == ["DK1"]


def test_outage_normalization_reads_nested_generation_units() -> None:
    events = build_outage_events(
        [
            {
                "messageId": "nested",
                "publicationDate": "2026-05-12T08:00:00Z",
                "unavailabilityReason": "Planned outage",
                "generationUnits": [
                    {
                        "areaName": "SE4",
                        "productionUnitName": "Plant A",
                        "name": "G1",
                        "timePeriods": [
                            {
                                "eventStart": "2026-05-10T00:00:00Z",
                                "eventStop": "2026-05-13T00:00:00Z",
                                "unavailableCapacity": 300,
                            }
                        ],
                    }
                ],
            }
        ]
    )
    assert events[0].zones == ["SE4"]
    assert events[0].title == "Planned outage: Plant A"
    assert events[0].time_start_utc == "2026-05-10T00:00:00Z"
    assert events[0].time_end_utc == "2026-05-13T00:00:00Z"
