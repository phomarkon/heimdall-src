from __future__ import annotations

import pytest

from tools.data.prepare_real_context import (
    ExtractStatus,
    _raise_for_required_failures,
    _resolve_window,
)


def test_prepare_real_context_fails_required_sources_only() -> None:
    statuses = [
        ExtractStatus("eds", "DK1", "Forecasts_Hour", 0, False, False, "optional missing"),
        ExtractStatus("entsoe_prices", "DK1", "day_ahead", 0, False, True, "required missing"),
    ]
    with pytest.raises(RuntimeError, match="required data extraction failed"):
        _raise_for_required_failures(statuses)


def test_prepare_real_context_allows_optional_degradation() -> None:
    _raise_for_required_failures(
        [ExtractStatus("eds", "DK1", "Forecasts_Hour", 0, False, False, "optional missing")]
    )


def test_prepare_real_context_resolves_explicit_utc_window() -> None:
    start, end = _resolve_window("2026-04-01T00:00:00Z", "2026-05-01T00:00:00Z", "2026-04")
    assert start.isoformat().startswith("2026-04-01T00:00:00+00:00")
    assert end.isoformat().startswith("2026-05-01T00:00:00+00:00")
