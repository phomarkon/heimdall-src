"""End-to-end POST /verify integration tests using the real trained F7 seed-42
checkpoint plus the ACI calibrator from its val residuals.

Skipped automatically when the F7 seed-42 checkpoint is missing (CI without
trained models). Per docs/RESEARCH-PROPOSAL.md §4.5 + §4.7.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from heimdall_verifier.calibrator import CalibratedForecaster
from heimdall_verifier.physical import default_p2h_spec, default_p2h_state
from heimdall_verifier.service import VerifyRequest, app

REPO_ROOT = Path(__file__).resolve().parents[3]
F7_DIR = REPO_ROOT / "models/forecaster/f7/seed-42"


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (F7_DIR / "model.pt").exists() or not (F7_DIR / "val_preds.npz").exists(),
        reason="F7 seed-42 checkpoint missing; run `experiments/seed_sweep.py` first.",
    ),
]


@pytest.fixture(scope="module")
def f7_model() -> tuple:
    """Load the F7 weights once for the whole module."""
    from heimdall_forecaster.train.model import PatchTransformerQuantile

    cfg = json.loads((F7_DIR / "config.json").read_text())
    model = PatchTransformerQuantile(
        n_features=int(cfg["n_features"]),
        seq_len=int(cfg["seq_len"]),
        horizon=int(cfg["horizon"]),
        n_quantiles=3,
        patch_len=int(cfg["patch_len"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        n_layers=int(cfg["n_layers"]),
        dropout=0.0,
    )
    state = torch.load(F7_DIR / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with open(F7_DIR / "stats.pkl", "rb") as fh:
        stats = pickle.load(fh)
    return model, stats


def _load_val_history() -> np.ndarray:
    """Last seq_len rows of the val target series, normalised to f7's stats."""
    import polars as pl

    df = pl.read_parquet(REPO_ROOT / "data/processed/dk1_panel_val.parquet")
    return df["imbalance_price_dkk_mwh_15min"].to_numpy().astype(np.float64)


def test_verify_accepts_safe_bid(f7_model) -> None:
    """A small bid well inside physical envelope and with a tight conformal
    interval should be accepted. End-to-end flow: F7 -> ACI -> verifier."""
    model, stats = f7_model
    cal = CalibratedForecaster.from_val_preds(F7_DIR / "val_preds.npz", alpha=0.1)
    series = _load_val_history()

    # Take the most recent seq_len chunk as history.
    seq_len = model.seq_len
    x = (series[-seq_len:] - stats.target_mean) / stats.target_std
    x = torch.from_numpy(x).float().reshape(1, seq_len, 1)
    with torch.no_grad():
        pred = model(x)  # (1, H, 3)
    q50 = stats.denormalise_target(pred[0, 0, 1].item())
    interval = cal.interval(point_pred=float(q50))

    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    spec_d = default_p2h_spec()
    state_d = default_p2h_state(now)

    # Pick a sell price comfortably above interval.upper so worst-case profit
    # is high. Bid 1 MW at upper + 10.
    payload = {
        "bid": {
            "market": "mFRR",
            "direction": "sell",
            "quantity_mw": 1.0,
            "price_eur_per_mwh": float(round(interval.upper + 10.0, 2)),
            "delivery_quarter": (now + timedelta(minutes=15)).isoformat(),
            "duration_minutes": 15,
        },
        "spec": {
            "q_max_mw": spec_d.q_max_mw,
            "ramp_mw_per_min": spec_d.ramp_mw_per_min,
            "storage_mwh": spec_d.storage_mwh,
            "cop": spec_d.cop,
            "loss_per_quarter": spec_d.loss_per_quarter,
            "bid_tick_eur": spec_d.bid_tick_eur,
        },
        "state": {
            "position_mw": state_d.position_mw,
            "last_delta_mw": state_d.last_delta_mw,
            "soc_mwh": state_d.soc_mwh,
            "cash_eur": state_d.cash_eur,
            "now_utc": now.isoformat(),
            "gate_closure_utc": (now - timedelta(minutes=5)).isoformat(),
        },
        "interval": {
            "horizon_minutes": 15,
            "alpha": 0.1,
            "lower": float(interval.lower),
            "upper": float(interval.upper),
            "method": "aci",
        },
        "tau_eur": -100.0,
    }
    client = TestClient(app)
    r = client.post("/verify", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] is True
    assert body["worst_case_profit_eur"] >= -100.0


def test_verify_rejects_ramp_violation(f7_model) -> None:
    """A bid that violates the ramp constraint must be rejected at the
    physical stage and surface the binding constraint."""
    model, stats = f7_model
    cal = CalibratedForecaster.from_val_preds(F7_DIR / "val_preds.npz", alpha=0.1)
    interval = cal.interval(point_pred=400.0)

    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    spec_d = default_p2h_spec()
    state_d = default_p2h_state(now)

    payload = {
        "bid": {
            "market": "mFRR",
            "direction": "sell",
            "quantity_mw": 50.0,  # ramp-rate is 5 MW/min × 15 min = 75 MW max ramp,
            "price_eur_per_mwh": float(round(interval.upper + 10.0, 2)),
            "delivery_quarter": (now + timedelta(minutes=15)).isoformat(),
            "duration_minutes": 15,
        },
        "spec": {
            "q_max_mw": 100.0,  # huge, so envelope doesn't bind
            "ramp_mw_per_min": 0.5,  # tiny ramp to force violation
            "storage_mwh": 1000.0,
            "cop": spec_d.cop,
            "loss_per_quarter": spec_d.loss_per_quarter,
            "bid_tick_eur": spec_d.bid_tick_eur,
        },
        "state": {
            "position_mw": state_d.position_mw,
            "last_delta_mw": 0.0,
            "soc_mwh": state_d.soc_mwh,
            "cash_eur": state_d.cash_eur,
            "now_utc": now.isoformat(),
            "gate_closure_utc": (now - timedelta(minutes=5)).isoformat(),
        },
        "interval": {
            "horizon_minutes": 15,
            "alpha": 0.1,
            "lower": float(interval.lower),
            "upper": float(interval.upper),
            "method": "aci",
        },
        "tau_eur": -100.0,
    }
    client = TestClient(app)
    r = client.post("/verify", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] is False
    assert body["stage_failed"] == "physical"
    assert body["physical_violation"]["constraint"] == "ramp_limit"


def test_verify_rejects_low_worst_case_profit(f7_model) -> None:
    """A bid that's physically feasible but whose price lies inside the
    conformal interval has worst-case profit ≤ 0 → must be rejected."""
    model, stats = f7_model
    cal = CalibratedForecaster.from_val_preds(F7_DIR / "val_preds.npz", alpha=0.1)
    series = _load_val_history()
    seq_len = model.seq_len
    x = (series[-seq_len:] - stats.target_mean) / stats.target_std
    x = torch.from_numpy(x).float().reshape(1, seq_len, 1)
    with torch.no_grad():
        pred = model(x)
    q50 = stats.denormalise_target(pred[0, 0, 1].item())
    interval = cal.interval(point_pred=float(q50))

    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    spec_d = default_p2h_spec()
    state_d = default_p2h_state(now)

    # Sell price *below* interval.upper guarantees worst-case profit < tau.
    bid_price = round(interval.lower + 1.0, 2)
    payload = {
        "bid": {
            "market": "mFRR",
            "direction": "sell",
            "quantity_mw": 1.0,
            "price_eur_per_mwh": float(bid_price),
            "delivery_quarter": (now + timedelta(minutes=15)).isoformat(),
            "duration_minutes": 15,
        },
        "spec": {
            "q_max_mw": spec_d.q_max_mw,
            "ramp_mw_per_min": spec_d.ramp_mw_per_min,
            "storage_mwh": spec_d.storage_mwh,
            "cop": spec_d.cop,
            "loss_per_quarter": spec_d.loss_per_quarter,
            "bid_tick_eur": spec_d.bid_tick_eur,
        },
        "state": {
            "position_mw": state_d.position_mw,
            "last_delta_mw": state_d.last_delta_mw,
            "soc_mwh": state_d.soc_mwh,
            "cash_eur": state_d.cash_eur,
            "now_utc": now.isoformat(),
            "gate_closure_utc": (now - timedelta(minutes=5)).isoformat(),
        },
        "interval": {
            "horizon_minutes": 15,
            "alpha": 0.1,
            "lower": float(interval.lower),
            "upper": float(interval.upper),
            "method": "aci",
        },
        "tau_eur": 100.0,  # require >= 100 EUR worst-case profit
    }
    client = TestClient(app)
    r = client.post("/verify", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] is False
    assert body["stage_failed"] == "conformal"
