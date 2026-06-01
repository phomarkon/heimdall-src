"""ENTSO-E client tests — fully mocked, NO live API calls."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from heimdall_data.entsoe import (
    EntsoeClient,
    _ensure_utc_index,
)


def test_ensure_utc_index_handles_naive_index() -> None:
    s = pd.Series([1.0, 2.0], index=pd.date_range("2025-03-04", periods=2, freq="15min"))
    out = _ensure_utc_index(s)
    assert str(out.index.tz) == "UTC"
    assert out.index.name == "timestamp_utc"


def test_token_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENTSOE_API_TOKEN", raising=False)
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ENTSO-E API token"):
        EntsoeClient()
