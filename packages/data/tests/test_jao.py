from __future__ import annotations

from datetime import UTC, datetime

from heimdall_data.jao import JAOClient, normalize_constraint_rows


class _Response:
    status_code = 200
    text = "ok"

    def json(self):
        return {"data": [{"timestamp": "2026-04-01T00:00:00Z", "publicationTime": "2026-03-31T23:50:00Z", "cnecId": "c1", "ram": 120.5, "shadowPrice": 44.0}]}


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, *, params, headers, timeout):  # type: ignore[no-untyped-def]
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        return _Response()


def test_jao_client_uses_auth_api_key_header() -> None:
    session = _Session()
    client = JAOClient(api_key="secret", session=session, live_fetch_enabled=True)

    rows = client.fetch_constraints(
        start=datetime(2026, 4, 1, tzinfo=UTC),
        end=datetime(2026, 4, 2, tzinfo=UTC),
        zone="DK1",
    )

    assert rows
    assert session.calls[0]["headers"] == {"AUTH_API_KEY": "secret"}


def test_jao_normalizes_constraint_rows() -> None:
    frame = normalize_constraint_rows(
        [
            {
                "deliveryTimestamp": "2026-04-01T00:00:00Z",
                "publicationTimestamp": "2026-03-31T23:45:00Z",
                "biddingZone": "DK1",
                "constraintId": "cnec-1",
                "constraintName": "DK1-DE limit",
                "remainingAvailableMargin": "300.25",
                "shadowPriceEurMw": "19.5",
                "commercialFlow": "-120",
                "fromZone": "DK1",
                "toZone": "DE",
            }
        ],
        zone="DK1",
    )

    row = frame.iloc[0]
    assert row["cnec_id"] == "cnec-1"
    assert row["ram_mw"] == 300.25
    assert row["shadow_price_eur_mw"] == 19.5
    assert row["direction"] == "DK1>DE"
