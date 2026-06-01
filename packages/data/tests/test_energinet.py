"""Energinet client tests — pytest-httpx mocks the API; NO live calls."""

from __future__ import annotations

import json

import pytest

from heimdall_data.energinet import BASE_URL, EnerginetClient, EnerginetError


def test_fetch_paginates_until_empty(httpx_mock) -> None:  # type: ignore[no-untyped-def]
    class _FakeResp:
        def __init__(self, payload: dict, status: int = 200) -> None:
            self._payload = payload
            self.status_code = status
            self.text = json.dumps(payload)

        def json(self) -> dict:
            return self._payload

    pages = [
        {"records": [{"HourUTC": "2025-03-04T00:00:00", "ImbalancePriceDKK": 100.0}] * 3},
        {"records": [{"HourUTC": "2025-03-04T00:45:00", "ImbalancePriceDKK": 99.0}]},
    ]
    calls: list[dict] = []

    class _FakeSession:
        def get(self, url: str, params: dict, timeout: float) -> _FakeResp:
            calls.append({"url": url, "params": dict(params)})
            idx = params["offset"] // params["limit"]
            return _FakeResp(pages[idx])

    c = EnerginetClient(session=_FakeSession())  # type: ignore[arg-type]
    rows = c.fetch("RegulatingBalancePowerdata", "2025-03-04T00:00", "2025-03-04T01:00", limit=3)
    assert len(rows) == 4
    assert calls[0]["url"] == f"{BASE_URL}/RegulatingBalancePowerdata"
    assert len(calls) == 2


def test_fetch_raises_on_4xx() -> None:
    class _FakeResp:
        status_code = 404
        text = "not found"

    class _FakeSession:
        def get(self, *a, **kw):  # type: ignore[no-untyped-def]
            return _FakeResp()

    c = EnerginetClient(session=_FakeSession())  # type: ignore[arg-type]
    with pytest.raises(EnerginetError, match="404"):
        c.fetch("Nope", "2025-03-04T00:00", "2025-03-05T00:00")
