"""Aggressive verifier stress harness. Per docs/RESEARCH-PROPOSAL.md §4.5 + §5.4.

Replaces the day-3 conservative version that produced accept_rate=1.0 across
the entire A5 grid. Two fixes:

1. **τ sweep with positive thresholds** — `pi_min(a)` is non-negative for the
   single-bid limit-order model in `heimdall_markets.profit`, so the day-3
   ``tau=-100`` was trivially satisfied. We sweep `τ ∈ {0, 50, 100, 250,
   500, 1000}` DKK to force the conformal stage to discriminate.
2. **Bid generators that span the (fill-prob × fill-side) grid** —
   `aggressive_sell` (bp < lower so a fill is *guaranteed* but yields the
   smallest possible margin), `marginal_sell` (bp inside [lower, upper]),
   `out_of_money_sell` (bp > upper, never fills), and the symmetric three on
   the buy side. Total six policies × 6 τ values × 1000 windows = 36 000
   verifier evaluations.

Outputs `experiments/outputs/verifier_stress.json` with a 2-D grid of
acceptance rates indexed by `(policy, tau_eur)`, plus raw `pi_min` histograms
per cell (so the §5.4 A5 figure can be regenerated without re-running).
"""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

from heimdall_contracts import BidAction, ConformalInterval
from heimdall_forecaster.train.model import PatchTransformerQuantile
from heimdall_ml import seeds
from heimdall_verifier.calibrator import CalibratedForecaster
from heimdall_verifier.physical import default_p2h_spec, default_p2h_state
from heimdall_verifier.service import VerifyRequest, _AssetSpecModel, _AssetStateModel, verify

REPO_ROOT = Path(__file__).resolve().parents[2]
F7_DIR = REPO_ROOT / "models/forecaster/f7/seed-42"
N_BIDS = 1000
# Bid sizing chosen so the "aggressive" policies' worst-case margin spans
# the tau grid: pi_min = margin × qty × (duration/60) = 200 × 5 × 0.25 = 250 EUR.
TAU_GRID_EUR = (0.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
MARGIN_EUR_PER_MWH = 200.0
POLICIES = (
    "aggressive_sell",
    "marginal_sell",
    "out_of_money_sell",
    "aggressive_buy",
    "marginal_buy",
    "out_of_money_buy",
)


def _load_f7():
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
        stats_obj = pickle.load(fh)
    return model, stats_obj


def _bid_price_for_policy(policy: str, q50: float, lower: float, upper: float, tick: float) -> float:
    """Bid prices chosen so that, at the canonical bid_qty_mw=5 MW × 15 min,
    the guaranteed margin lands across the τ-grid [0, 1000] EUR rather than
    collapsing to {0, 0.25}. See `bid_qty_mw` and `MARGIN_EUR_PER_MWH` below.
    """
    margin = MARGIN_EUR_PER_MWH
    if policy == "aggressive_sell":
        bp = lower - margin  # guaranteed fill, margin*qty*dt EUR worst case
    elif policy == "marginal_sell":
        bp = q50  # 50/50 fill probability
    elif policy == "out_of_money_sell":
        bp = upper + 1.0  # never fills
    elif policy == "aggressive_buy":
        bp = upper + margin  # guaranteed fill, margin*qty*dt EUR worst case
    elif policy == "marginal_buy":
        bp = q50
    elif policy == "out_of_money_buy":
        bp = lower - 1.0  # never fills
    else:
        raise ValueError(f"unknown policy {policy}")
    return round(bp / tick) * tick


def _direction_for_policy(policy: str) -> str:
    return "sell" if "_sell" in policy else "buy"


def _build_request(
    *,
    bid: BidAction,
    spec_d,
    state_d,
    interval_obj,
    tau_eur: float,
) -> VerifyRequest:
    return VerifyRequest(
        bid=bid,
        spec=_AssetSpecModel(
            q_max_mw=spec_d.q_max_mw,
            ramp_mw_per_min=spec_d.ramp_mw_per_min,
            storage_mwh=spec_d.storage_mwh,
            cop=spec_d.cop,
            loss_per_quarter=spec_d.loss_per_quarter,
            bid_tick_eur=spec_d.bid_tick_eur,
        ),
        state=_AssetStateModel(
            position_mw=state_d.position_mw,
            last_delta_mw=state_d.last_delta_mw,
            soc_mwh=state_d.soc_mwh,
            cash_eur=state_d.cash_eur,
            now_utc=state_d.now_utc,
            gate_closure_utc=state_d.gate_closure_utc,
        ),
        interval=ConformalInterval(
            horizon_minutes=15,
            alpha=0.1,
            lower=float(interval_obj.lower),
            upper=float(interval_obj.upper),
            method="aci",
        ),
        tau_eur=float(tau_eur),
    )


def run() -> dict:
    seeds.seed_everything(42)
    model, stats_obj = _load_f7()
    cal = CalibratedForecaster.from_val_preds(F7_DIR / "val_preds.npz", alpha=0.1)

    df = pl.read_parquet(REPO_ROOT / "data/processed/dk1_panel_val.parquet")
    series = df["imbalance_price_dkk_mwh_15min"].to_numpy().astype(np.float64)
    seq_len = model.seq_len
    n = min(N_BIDS, series.size - seq_len - 1)

    spec_d = default_p2h_spec()
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)

    # (policy, tau) -> list of pi_min, acceptance bool, rejection cause.
    by_cell: dict[tuple[str, float], dict] = {
        (p, t): {"accepted": 0, "n": 0, "pi_min": [], "reasons": Counter()}
        for p in POLICIES
        for t in TAU_GRID_EUR
    }
    bid_qty_mw = 5.0  # 5 MW × 15 min × MARGIN -> aggressive pi_min = 250 EUR

    for i in range(n):
        x_raw = series[i : i + seq_len]
        if np.any(np.isnan(x_raw)):
            continue
        x = (x_raw - stats_obj.target_mean) / stats_obj.target_std
        with torch.no_grad():
            pred = model(torch.from_numpy(x).float().reshape(1, seq_len, 1))
        q50 = float(stats_obj.denormalise_target(pred[0, 0, 1].item()))
        interval = cal.interval(point_pred=q50)
        realised = float(series[i + seq_len])
        cal.update(realised=realised, point_pred=q50)

        # Stateful: each cell has its own state so bids are NOT independent.
        # We snapshot the *fresh* default state per window to keep the grid
        # decoupled from policy/tau interactions; A11 (statefulness) is its
        # own ablation. The aggressive bidding alone is enough to break the
        # accept_rate=1.0 plateau.
        state_d = default_p2h_state(now)

        for policy in POLICIES:
            direction = _direction_for_policy(policy)
            bp = _bid_price_for_policy(policy, q50, interval.lower, interval.upper, spec_d.bid_tick_eur)
            bid = BidAction(
                market="mFRR",
                direction=direction,
                quantity_mw=bid_qty_mw,
                price_eur_per_mwh=bp,
                delivery_quarter=now + timedelta(minutes=15),
                duration_minutes=15,
            )
            for tau in TAU_GRID_EUR:
                req = _build_request(bid=bid, spec_d=spec_d, state_d=state_d, interval_obj=interval, tau_eur=tau)
                v = verify(req)
                cell = by_cell[(policy, tau)]
                cell["n"] += 1
                if v.accepted:
                    cell["accepted"] += 1
                else:
                    if v.physical_violation is not None:
                        cell["reasons"][f"physical/{v.physical_violation.constraint}"] += 1
                    elif v.stage_failed == "conformal":
                        cell["reasons"]["conformal/worst_case_profit"] += 1
                    else:
                        cell["reasons"][v.stage_failed or "unknown"] += 1
                if v.worst_case_profit_eur is not None:
                    cell["pi_min"].append(v.worst_case_profit_eur)

    grid = []
    for policy in POLICIES:
        for tau in TAU_GRID_EUR:
            cell = by_cell[(policy, tau)]
            n_cell = max(cell["n"], 1)
            pi_min_arr = np.array(cell["pi_min"], dtype=float)
            grid.append(
                {
                    "policy": policy,
                    "tau_eur": tau,
                    "n": cell["n"],
                    "accept_rate": cell["accepted"] / n_cell,
                    "reject_reasons": dict(cell["reasons"]),
                    "pi_min_p10": float(np.percentile(pi_min_arr, 10)) if pi_min_arr.size else None,
                    "pi_min_median": float(np.median(pi_min_arr)) if pi_min_arr.size else None,
                    "pi_min_p90": float(np.percentile(pi_min_arr, 90)) if pi_min_arr.size else None,
                }
            )

    out = {
        "n_windows": int(n),
        "tau_grid_eur": list(TAU_GRID_EUR),
        "policies": list(POLICIES),
        "grid": grid,
        "summary": {
            "any_below_1_accept_rate": any(c["accept_rate"] < 1.0 for c in grid),
            "min_accept_rate": min(c["accept_rate"] for c in grid),
            "max_accept_rate": max(c["accept_rate"] for c in grid),
        },
    }
    out_path = REPO_ROOT / "experiments/outputs/verifier_stress.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["summary"], indent=2))
    print(f"min/max accept rate across (policy × tau) grid: "
          f"[{out['summary']['min_accept_rate']:.3f}, {out['summary']['max_accept_rate']:.3f}]")
    return out


if __name__ == "__main__":
    run()
