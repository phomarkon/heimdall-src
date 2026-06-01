from __future__ import annotations

import pandas as pd
import pytest
from heimdall_data.entsoe import normalize_activation_frame

from tools.data.prepare_evaluation_truth import build_activation_truth


def test_normalize_activation_frame_extracts_up_down_volumes() -> None:
    raw = pd.DataFrame(
        {
            "Activated upward mFRR": [4.0, 0.0],
            "Activated downward mFRR": [0.0, 3.0],
            "A96 metadata only": [99.0, 99.0],
        },
        index=pd.to_datetime(["2026-04-01T00:00:00Z", "2026-04-01T00:15:00Z"]),
    )

    frame = normalize_activation_frame(raw, zone="DK1")

    assert frame.to_dict(orient="records") == [
        {
            "timestamp_utc": pd.Timestamp("2026-04-01T00:00:00Z"),
            "zone": "DK1",
            "activation_direction": "up",
            "activated_volume_mwh": 4.0,
        },
        {
            "timestamp_utc": pd.Timestamp("2026-04-01T00:15:00Z"),
            "zone": "DK1",
            "activation_direction": "down",
            "activated_volume_mwh": 3.0,
        },
    ]


def test_build_activation_truth_requires_activation_volume_unless_price_only_flag() -> None:
    activations = pd.DataFrame(columns=["timestamp_utc", "zone", "activation_direction", "activated_volume_mwh"])
    prices = pd.DataFrame(
        [
            {
                "timestamp_utc": pd.Timestamp("2026-04-01T00:00:00Z"),
                "zone": "DK1",
                "price_type": "day_ahead",
                "price_eur_mwh": 50.0,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="activation truth is required"):
        build_activation_truth(activations, prices)

    truth = build_activation_truth(activations, prices, allow_price_only_diagnostics=True)

    assert truth["activation_direction"].tolist() == ["unknown"]
    assert truth["spot_price_eur_mwh"].tolist() == [50.0]
