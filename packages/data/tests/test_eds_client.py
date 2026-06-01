from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from heimdall_data.eds import EDSClient, normalize_eds_table


def test_eds_client_paginates_with_bounded_limit_and_throttle(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    sleeps: list[float] = []

    def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("heimdall_data.eds.time.sleep", record_sleep)

    class _Response:
        status_code = 200
        text = "{}"

        def __init__(self, rows: list[dict]) -> None:
            self._rows = rows
            self.headers = {}

        def json(self) -> dict:
            return {"records": self._rows}

    calls: list[dict] = []

    class _Session:
        def get(self, _url: str, *, params: dict, timeout: float) -> _Response:
            calls.append(dict(params))
            if params["offset"] == 0:
                return _Response([{"TimeUTC": "2026-04-01T00:00:00Z", "PriceArea": "DK1"}] * 2)
            return _Response([{"TimeUTC": "2026-04-01T00:30:00Z", "PriceArea": "DK1"}])

    client = EDSClient(session=_Session(), page_size=2, delay_seconds=0.5)  # type: ignore[arg-type]
    frame = client.fetch_dataset(
        "Elspotprices",
        start=datetime(2026, 4, 1, tzinfo=UTC),
        end=datetime(2026, 4, 2, tzinfo=UTC),
        filters={"PriceArea": "DK1"},
    )
    assert len(frame) == 3
    assert [call["limit"] for call in calls] == [2, 2]
    assert all(call["limit"] != 0 for call in calls)
    assert calls[1]["offset"] == 2
    assert sleeps


def test_normalize_eds_table_maps_time_and_zone() -> None:
    out = normalize_eds_table(
        pd.DataFrame(
            {
                "HourUTC": ["2026-04-01T00:00:00Z"],
                "PriceArea": ["DK1"],
                "Value": ["12.5"],
            }
        ),
        dataset="Forecasts_Hour",
    )
    assert out.loc[0, "zone"] == "DK1"
    assert float(out.loc[0, "Value"]) == 12.5
    assert str(out.loc[0, "timestamp_utc"].tz) == "UTC"
