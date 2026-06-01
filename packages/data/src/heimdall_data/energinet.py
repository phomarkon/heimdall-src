"""Energinet Open Data API wrapper.

Per docs/RESEARCH-PROPOSAL.md §5.1, Energinet is the DK-native source — finer-grained
mFRR + imbalance data than ENTSO-E for DK1/DK2 and no API key required. JSON+REST.

Datasets we read (full list at https://www.energidataservice.dk/):
  - ``RegulatingBalancePowerdata`` — imbalance settlement prices, regulation
    up/down, mFRR activated up/down (single dataset covers both signals on
    the live API as of 2026-05). Probed on 2026-05-09.

The endpoint is `https://api.energidataservice.dk/dataset/{name}`. It returns
paginated JSON; we follow ``next`` until exhausted with bounded retries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

BASE_URL = "https://api.energidataservice.dk/dataset"
DEFAULT_TIMEOUT_S = 30.0


class EnerginetError(RuntimeError):
    """Raised when the Energinet API returns a non-2xx or malformed payload."""


@dataclass
class EnerginetClient:
    """Stateless client. ``session`` is overridable for tests."""

    session: requests.Session | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        reraise=True,
    )
    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self.session is not None
        r = self.session.get(url, params=params, timeout=self.timeout_s)
        if r.status_code >= 400:
            raise EnerginetError(f"GET {url} -> {r.status_code}: {r.text[:200]}")
        try:
            return r.json()
        except ValueError as e:  # pragma: no cover -- malformed JSON
            raise EnerginetError(f"non-json from {url}: {e}") from e

    def fetch(
        self,
        dataset: str,
        start: str,
        end: str,
        *,
        filter_json: str | None = None,
        limit: int = 100_000,
    ) -> list[dict[str, Any]]:
        """Fetch all rows in [start, end) for ``dataset``.

        Times are ISO 8601, e.g. ``"2025-03-04T00:00"``. ``filter_json`` is the
        Energinet-specific URL filter, e.g. ``'{"PriceArea":"DK1"}'``.
        """
        params: dict[str, Any] = {
            "start": start,
            "end": end,
            "limit": limit,
            "offset": 0,
        }
        if filter_json:
            params["filter"] = filter_json
        url = f"{BASE_URL}/{dataset}"
        rows: list[dict[str, Any]] = []
        while True:
            payload = self._get(url, params)
            batch = payload.get("records", [])
            rows.extend(batch)
            if len(batch) < limit:
                break
            params["offset"] += limit
        return rows

    # --- convenience wrappers -------------------------------------------

    def regulating_balance(self, start: str, end: str, area: str = "DK1") -> list[dict[str, Any]]:
        """RegulatingBalancePowerdata — imbalance prices + mFRR Up/Down activated.

        Schema (subset, probed 2026-05-09 on the live API):
          ``HourUTC``, ``PriceArea``,
          ``ImbalancePriceDKK``, ``ImbalancePriceEUR``,
          ``mFRRUpActBal``  (mFRR up activated, MWh),
          ``mFRRDownActBal`` (mFRR down activated, MWh),
          ``BalancingPowerPriceUpEUR``, ``BalancingPowerPriceDownEUR``.
        """
        return self.fetch(
            "RegulatingBalancePowerdata", start, end, filter_json=f'{{"PriceArea":"{area}"}}'
        )

    # --- new endpoints (2026-05-10) -----------------------------------------

    def forecasts_hour(self, start: str, end: str, area: str = "DK1") -> list[dict[str, Any]]:
        """``Forecasts_Hour`` — DA wind onshore/offshore + solar forecasts (MW).

        Free, no auth.  Schema (subset, current API):
          ``HourUTC``, ``PriceArea``, ``ForecastType``, ``ForecastDayAhead``,
          ``ForecastIntraday``, ``Forecast5Hour``, ``ForecastCurrent``.
        ``ForecastType`` enumerates ``Solar``, ``Onshore Wind``, ``Offshore Wind``.
        """
        return self.fetch(
            "Forecasts_Hour", start, end, filter_json=f'{{"PriceArea":"{area}"}}'
        )

    def production_consumption_settlement(self, start: str, end: str, area: str = "DK1") -> list[dict[str, Any]]:
        """``ProductionConsumptionSettlement`` — actual generation by source (MWh per hour).

        Schema highlights:
          ``HourUTC``, ``PriceArea``,
          ``OffshoreWindGe50``  (offshore wind ≥50 MW),
          ``OffshoreWindLt50``,
          ``OnshoreWindGe50``,
          ``OnshoreWindLt50``,
          ``SolarPowerGe40``,
          ``SolarPowerLt40``,
          ``GrossConsumptionMWh``,
          ``ExchangeNO``, ``ExchangeSE``, ``ExchangeGE``, ``ExchangeNL``.
        """
        return self.fetch(
            "ProductionConsumptionSettlement", start, end, filter_json=f'{{"PriceArea":"{area}"}}'
        )

    def elspot_prices(self, start: str, end: str, area: str = "DK1") -> list[dict[str, Any]]:
        """``Elspotprices`` — Nord Pool day-ahead clearing prices (EUR/MWh, DKK/MWh)."""
        return self.fetch(
            "Elspotprices", start, end, filter_json=f'{{"PriceArea":"{area}"}}'
        )

    def co2_emis(self, start: str, end: str, area: str = "DK1") -> list[dict[str, Any]]:
        """``CO2Emis`` — DK grid carbon-intensity 5-minute series (g CO₂eq/kWh)."""
        return self.fetch(
            "CO2Emis", start, end, filter_json=f'{{"PriceArea":"{area}"}}'
        )


__all__ = ["BASE_URL", "EnerginetClient", "EnerginetError"]
