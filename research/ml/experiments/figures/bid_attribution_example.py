"""Worked example: bid attribution combining (i) physical binding-constraint
and (ii) conformal-tail driver (top SHAP feature on q50).

Builds a synthetic but realistic verifier scenario, runs ``attribute_*``,
renders both pieces side-by-side. Saves
``figures/bid_attribution_example.png``.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from heimdall_contracts import BidAction
from heimdall_forecaster.train.dataset import F8_FEATURES, SEQ_LEN
from heimdall_forecaster.train.model import PatchTransformerQuantile
from heimdall_ml.explain.bid_attribution import (
    attribute_conformal_tail,
    attribute_verifier_verdict,
)
from heimdall_ml.viz import apply_paper_style, paper_palette
from heimdall_ml.viz.style import PAGE_WIDTH_IN
from heimdall_verifier.calibrator import CalibratedForecaster
from heimdall_verifier.physical import default_p2h_spec, default_p2h_state
from heimdall_verifier.service import VerifyRequest, _AssetSpecModel, _AssetStateModel, verify

REPO_ROOT = Path(__file__).resolve().parents[3]
F8_DIR = REPO_ROOT / "models/forecaster/f8/seed-42"
FIG_DIR = REPO_ROOT / "figures"


def _load_f8():
    cfg = json.loads((F8_DIR / "config.json").read_text())
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
    state = torch.load(F8_DIR / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with open(F8_DIR / "stats.pkl", "rb") as fh:
        stats = pickle.load(fh)
    return model, stats


def main() -> int:
    apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model, stats = _load_f8()
    cal = CalibratedForecaster.from_val_preds(F8_DIR / "val_preds.npz", alpha=0.1)

    df = pl.read_parquet(REPO_ROOT / "data/processed/dk1_panel_val.parquet").drop_nulls()
    arr = df.select(F8_FEATURES).to_numpy().astype(np.float64)
    arr_norm = stats.normalise(arr)
    # window ending at row 1000
    end = 1000
    x_window = arr_norm[end - SEQ_LEN : end].astype(np.float32)

    # base forecast
    with torch.no_grad():
        pred = model(torch.from_numpy(x_window[None]).float())
    q50 = float(stats.denormalise_target(pred[0, 0, 1].item()))
    interval = cal.interval(point_pred=q50)

    # Build a SELL bid that will fail conformal at tau=200 (likely).
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    bid_price = round((q50) / 0.01) * 0.01
    bid = BidAction(
        market="mFRR",
        direction="sell",
        quantity_mw=1.0,
        price_eur_per_mwh=bid_price,
        delivery_quarter=now + timedelta(minutes=15),
        duration_minutes=15,
    )
    spec_d = default_p2h_spec()
    state_d = default_p2h_state(now)
    req = VerifyRequest(
        bid=bid,
        spec=_AssetSpecModel(
            q_max_mw=spec_d.q_max_mw, ramp_mw_per_min=spec_d.ramp_mw_per_min,
            storage_mwh=spec_d.storage_mwh, cop=spec_d.cop,
            loss_per_quarter=spec_d.loss_per_quarter, bid_tick_eur=spec_d.bid_tick_eur,
        ),
        state=_AssetStateModel(
            position_mw=state_d.position_mw, last_delta_mw=state_d.last_delta_mw,
            soc_mwh=state_d.soc_mwh, cash_eur=state_d.cash_eur,
            now_utc=state_d.now_utc, gate_closure_utc=now - timedelta(minutes=5),
        ),
        interval=interval,
        tau_eur=200.0,
    )
    verdict = verify(req)
    physical_attrib = attribute_verifier_verdict(verdict)

    # Now the conformal-tail attribution: perturb the most recent step of each
    # feature by ±1σ and read off the change in pi_min.
    sigma = stats.std
    # Only the last step matters for a 1-sigma perturbation here; we collapse
    # by perturbing the entire window's last step for a given feature.
    base_last = x_window[-1].copy()

    def _run(features_last: np.ndarray):
        x = x_window.copy()
        x[-1] = features_last
        with torch.no_grad():
            pred = model(torch.from_numpy(x[None]).float())
        q50_loc = float(stats.denormalise_target(pred[0, 0, 1].item()))
        return cal.interval(point_pred=q50_loc)

    # Use train sigma in normalised units (since features_last is normalised).
    # Per-feature normalised sigma is 1.0 by construction; we perturb by 1.0.
    sensitivities = attribute_conformal_tail(
        bid=bid,
        base_features=base_last,
        feature_names=F8_FEATURES,
        feature_sigma=np.ones(len(F8_FEATURES)),
        forecaster_to_interval=_run,
    )

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 2.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.5)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Left panel: physical attribution as text + the verdict tag.
    ax_a.axis("off")
    ax_a.set_title("(a) Verifier verdict", fontsize=8)
    lines = [
        f"accepted: {verdict.accepted}",
        f"stage_failed: {verdict.stage_failed or '—'}",
        f"binding: {physical_attrib.get('binding_constraint') or '—'}",
        f"slack: {physical_attrib.get('slack')!s}",
        f"q50 = {q50:.1f} DKK/MWh",
        f"interval = [{interval.lower:.1f}, {interval.upper:.1f}]",
        f"τ = {req.tau_eur:.1f}",
    ]
    ax_a.text(0.0, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=7)

    # Right panel: top conformal-tail sensitivities.
    palette = paper_palette(len(sensitivities))
    names = [s.feature for s in sensitivities]
    plus = [s.delta_pi_plus for s in sensitivities]
    minus = [s.delta_pi_minus for s in sensitivities]
    y = np.arange(len(names))
    ax_b.barh(y - 0.18, plus, height=0.36, color=palette[0], label="+1σ")
    ax_b.barh(y + 0.18, minus, height=0.36, color=palette[1], label="−1σ")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7)
    ax_b.axvline(0, color="k", lw=0.5)
    ax_b.set_xlabel("Δπ_min (EUR)")
    ax_b.set_title("(b) Conformal-tail sensitivity", fontsize=8)
    ax_b.legend(fontsize=7, loc="lower right")

    fig.savefig(FIG_DIR / "bid_attribution_example.png")
    plt.close(fig)
    print(f"-> {FIG_DIR / 'bid_attribution_example.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
