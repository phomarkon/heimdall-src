from __future__ import annotations

import math
import os

import pytest

from packages.data import fetch_eds_dataset, normalize_eds_imbalance_price
from packages.simulator.mfrr_engine import backtest_price_model

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_HEIMDALL_INTEGRATION") != "1",
    reason="set RUN_HEIMDALL_INTEGRATION=1 to run live EDS mFRR engine validation",
)
def test_live_eds_mfrr_engine_backtest_covers_both_danish_zones() -> None:
    raw = fetch_eds_dataset(
        "ImbalancePrice",
        start="2025-03-04",
        end="2025-03-18",
        price_areas=["DK1", "DK2"],
    )
    normalized = normalize_eds_imbalance_price(raw)

    report = backtest_price_model(normalized, train_fraction=0.7, min_samples=20)

    assert set(report.zone_metrics) == {"DK1", "DK2"}
    assert report.total_rows > 100
    assert math.isfinite(report.mae_eur_mwh)
    assert math.isfinite(report.max_abs_error_eur_mwh)
    assert 0.0 <= report.interval_coverage_90 <= 1.0
    assert report.result_hash
