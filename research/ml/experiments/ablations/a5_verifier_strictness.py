"""A5 — verifier strictness (alpha × tau grid). docs/RESEARCH-PROPOSAL.md §5.4.

Reuses the aggressive 6-policy bid generator from `verifier_stress.py` and adds
an alpha dimension. For each α ∈ {0.01, 0.05, 0.10, 0.20} we sweep the full
(policy × τ) grid. This shows:

1. Aggressive policies (sell at lower - margin, buy at upper + margin):
   pi_min = margin × qty × dt ≈ 250 EUR independent of α — the worst case
   always lands on a filling endpoint where margin dominates.

2. Marginal policies (bid at q50): pi_min = 0 independent of α — the non-
   filling endpoint always provides the zero-profit floor.

3. Alpha's effect on acceptance rate is therefore indirect: wider intervals
   (smaller α) shift the bid prices for aggressive policies, which can
   trigger different physical-constraint outcomes (cash floor, ramp, etc.).

This replaces the old tautological version (sell-only, negative tau).
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

REPO_ROOT = Path(__file__).resolve().parents[3]
F7_DIR = REPO_ROOT / "models/forecaster/f7/seed-42"
N_BIDS = 1000
TAU_GRID_EUR = (0.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
ALPHA_GRID = (0.01, 0.05, 0.10, 0.20)
POLICIES = (
    "aggressive_sell",
    "marginal_sell",
    "out_of_money_sell",
    "aggressive_buy",
    "marginal_buy",
    "out_of_money_buy",
)
MARGIN_EUR_PER_MWH = 200.0
BID_QTY_MW = 5.0


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


def _bid_price(policy: str, q50: float, lower: float, upper: float, tick: float) -> float:
    margin = MARGIN_EUR_PER_MWH
    if policy == "aggressive_sell":
        bp = lower - margin
    elif policy == "marginal_sell":
        bp = q50
    elif policy == "out_of_money_sell":
        bp = upper + 1.0
    elif policy == "aggressive_buy":
        bp = upper + margin
    elif policy == "marginal_buy":
        bp = q50
    elif policy == "out_of_money_buy":
        bp = lower - 1.0
    else:
        raise ValueError(f"unknown policy {policy}")
    return round(bp / tick) * tick


def _direction(policy: str) -> str:
    return "sell" if "_sell" in policy else "buy"


def run() -> dict:
    seeds.seed_everything(42)
    model, stats_obj = _load_f7()
    df = pl.read_parquet(REPO_ROOT / "data/processed/dk1_panel_val.parquet")
    series = df["imbalance_price_dkk_mwh_15min"].to_numpy().astype(np.float64)
    seq_len = model.seq_len
    n = min(N_BIDS, series.size - seq_len - 1)
    spec_d = default_p2h_spec()
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)

    # (alpha, policy, tau) -> accumulators
    by_cell: dict[tuple[float, str, float], dict] = defaultdict(
        lambda: {"accepted": 0, "n": 0, "pi_min": [], "reasons": Counter()}
    )

    # One calibrator per alpha (maintains independent online ACI state).
    cals = {alpha: CalibratedForecaster.from_val_preds(F7_DIR / "val_preds.npz", alpha=alpha)
            for alpha in ALPHA_GRID}

    for i in range(n):
        x_raw = series[i : i + seq_len]
        if np.any(np.isnan(x_raw)):
            continue
        x = (x_raw - stats_obj.target_mean) / stats_obj.target_std
        with torch.no_grad():
            pred = model(torch.from_numpy(x).float().reshape(1, seq_len, 1))
        q50 = float(stats_obj.denormalise_target(pred[0, 0, 1].item()))
        realised = float(series[i + seq_len])

        for alpha in ALPHA_GRID:
            cal = cals[alpha]
            interval = cal.interval(point_pred=q50)
            cal.update(realised=realised, point_pred=q50)

            state_d = default_p2h_state(now)
            for policy in POLICIES:
                bp = _bid_price(policy, q50, interval.lower, interval.upper, spec_d.bid_tick_eur)
                bid = BidAction(
                    market="mFRR",
                    direction=_direction(policy),
                    quantity_mw=BID_QTY_MW,
                    price_eur_per_mwh=bp,
                    delivery_quarter=now + timedelta(minutes=15),
                    duration_minutes=15,
                )
                for tau in TAU_GRID_EUR:
                    req = VerifyRequest(
                        bid=bid,
                        spec=_AssetSpecModel(**spec_d.__dict__),
                        state=_AssetStateModel(
                            position_mw=state_d.position_mw,
                            last_delta_mw=state_d.last_delta_mw,
                            soc_mwh=state_d.soc_mwh,
                            cash_eur=state_d.cash_eur,
                            now_utc=state_d.now_utc,
                            gate_closure_utc=now - timedelta(minutes=5),
                        ),
                        interval=ConformalInterval(
                            horizon_minutes=15, alpha=alpha,
                            lower=float(interval.lower), upper=float(interval.upper), method="aci",
                        ),
                        tau_eur=float(tau),
                    )
                    v = verify(req)
                    cell = by_cell[(alpha, policy, tau)]
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
    for (alpha, policy, tau), cell in sorted(by_cell.items()):
        n_cell = max(cell["n"], 1)
        arr = np.array(cell["pi_min"], dtype=float)
        grid.append({
            "alpha": alpha,
            "policy": policy,
            "tau_eur": tau,
            "n": cell["n"],
            "accept_rate": cell["accepted"] / n_cell,
            "reject_reasons": dict(cell["reasons"]),
            "pi_min_p10": float(np.percentile(arr, 10)) if arr.size else None,
            "pi_min_median": float(np.median(arr)) if arr.size else None,
            "pi_min_p90": float(np.percentile(arr, 90)) if arr.size else None,
        })

    out_path = REPO_ROOT / "experiments/outputs/a5_verifier_strictness.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(grid, indent=2))

    # Emit readable summary
    print(f"{'alpha':>6s} {'policy':>22s} " + " ".join(f"tau={t:>5.0f}" for t in TAU_GRID_EUR))
    print("-" * (6 + 1 + 22 + 1 + len(TAU_GRID_EUR) * 10))
    for alpha in ALPHA_GRID:
        for pol in POLICIES:
            row = f"{alpha:>6.2f} {pol:>22s}"
            for tau in TAU_GRID_EUR:
                c = next(c for c in grid if c["alpha"] == alpha and c["policy"] == pol and c["tau_eur"] == tau)
                row += f" {c['accept_rate']:>9.2f}"
            print(row)

    print(f"\nOutput → {out_path}")
    return {"grid": grid}


if __name__ == "__main__":
    run()
