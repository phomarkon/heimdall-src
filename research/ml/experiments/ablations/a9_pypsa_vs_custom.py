"""A9 — PyPSA-Eur-Sec-derived AssetSpec vs hand-coded `default_p2h_spec()`.

Per docs/RESEARCH-PROPOSAL.md §5.4 (A9): "validate that PyPSA-derived parameters
reproduce DK1 imbalance dynamics better than literature-mean defaults."

Minimal cell (per A9 review notes/findings/2026-05-09-pypsa-adapter-review.md):
re-run the verifier-stress harness from `experiments/verifier_stress.py` with
both AssetSpecs, compare:
  - acceptance-rate grid (policy × τ)
  - π_min_median per cell
  - rejection-reason distribution (which physical constraint binds first)

Output: `experiments/outputs/a9_pypsa_vs_custom.json` + a small markdown
summary at `notes/ablations/A9.md`.
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
from heimdall_verifier.physical import AssetSpec, default_p2h_spec, default_p2h_state
from heimdall_verifier.scenario_loader import (
    assetspec_from_pypsa_eursec_dk_network,
    assetspec_from_tiny_dk_network,
)
from heimdall_verifier.service import (
    VerifyRequest,
    _AssetSpecModel,
    _AssetStateModel,
    verify,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
F7_DIR = REPO_ROOT / "models/forecaster/f7/seed-42"
N_BIDS = 1000
TAU_GRID_EUR = (0.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
MARGIN_EUR_PER_MWH = 200.0
# Bid sizing.  Default = canonical 5 MW × 15 min (aligns with verifier_stress).
# A9-entrant variant (`--bid-qty 40`) surfaces ramp + SoC envelope differences.
DEFAULT_BID_QTY_MW = 5.0
DEFAULT_OUT_BASENAME = "a9_pypsa_vs_custom"
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


def _bid_price(policy: str, q50: float, lo: float, hi: float, tick: float) -> float:
    if policy == "aggressive_sell":
        bp = lo - MARGIN_EUR_PER_MWH
    elif policy == "marginal_sell":
        bp = q50
    elif policy == "out_of_money_sell":
        bp = hi + 1.0
    elif policy == "aggressive_buy":
        bp = hi + MARGIN_EUR_PER_MWH
    elif policy == "marginal_buy":
        bp = q50
    elif policy == "out_of_money_buy":
        bp = lo - 1.0
    else:
        raise ValueError(policy)
    return round(bp / tick) * tick


def _direction(policy: str) -> str:
    return "sell" if "_sell" in policy else "buy"


def _request(bid: BidAction, spec: AssetSpec, state, interval, tau: float) -> VerifyRequest:
    return VerifyRequest(
        bid=bid,
        spec=_AssetSpecModel(
            q_max_mw=spec.q_max_mw,
            ramp_mw_per_min=spec.ramp_mw_per_min,
            storage_mwh=spec.storage_mwh,
            cop=spec.cop,
            loss_per_quarter=spec.loss_per_quarter,
            bid_tick_eur=spec.bid_tick_eur,
        ),
        state=_AssetStateModel(
            position_mw=state.position_mw,
            last_delta_mw=state.last_delta_mw,
            soc_mwh=state.soc_mwh,
            cash_eur=state.cash_eur,
            now_utc=state.now_utc,
            gate_closure_utc=state.gate_closure_utc,
        ),
        interval=ConformalInterval(
            horizon_minutes=15,
            alpha=0.1,
            lower=float(interval.lower),
            upper=float(interval.upper),
            method="aci",
        ),
        tau_eur=float(tau),
    )


def _stress_one_spec(spec: AssetSpec, model, stats_obj, cal, series, seq_len, n,
                     bid_qty_mw: float = DEFAULT_BID_QTY_MW) -> dict:
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    by_cell = {
        (p, t): {"accepted": 0, "n": 0, "pi_min": [], "reasons": Counter()}
        for p in POLICIES
        for t in TAU_GRID_EUR
    }
    cal_local = cal  # ACI is stateful; we share so both specs see the same intervals

    for i in range(n):
        x_raw = series[i : i + seq_len]
        if np.any(np.isnan(x_raw)):
            continue
        x = (x_raw - stats_obj.target_mean) / stats_obj.target_std
        with torch.no_grad():
            pred = model(torch.from_numpy(x).float().reshape(1, seq_len, 1))
        q50 = float(stats_obj.denormalise_target(pred[0, 0, 1].item()))
        interval = cal_local.interval(point_pred=q50)
        # Note: do NOT update ACI here so the second spec sees the same path.
        state_d = default_p2h_state(now)
        for policy in POLICIES:
            d = _direction(policy)
            bp = _bid_price(policy, q50, interval.lower, interval.upper, spec.bid_tick_eur)
            bid = BidAction(
                market="mFRR",
                direction=d,
                quantity_mw=bid_qty_mw,
                price_eur_per_mwh=bp,
                delivery_quarter=now + timedelta(minutes=15),
                duration_minutes=15,
            )
            for tau in TAU_GRID_EUR:
                v = verify(_request(bid, spec, state_d, interval, tau))
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
    for p in POLICIES:
        for t in TAU_GRID_EUR:
            cell = by_cell[(p, t)]
            n_cell = max(cell["n"], 1)
            arr = np.array(cell["pi_min"], dtype=float)
            grid.append(
                {
                    "policy": p,
                    "tau_eur": t,
                    "n": cell["n"],
                    "accept_rate": cell["accepted"] / n_cell,
                    "reject_reasons": dict(cell["reasons"]),
                    "pi_min_p10": float(np.percentile(arr, 10)) if arr.size else None,
                    "pi_min_median": float(np.median(arr)) if arr.size else None,
                    "pi_min_p90": float(np.percentile(arr, 90)) if arr.size else None,
                }
            )
    return {"grid": grid, "spec": spec.__dict__}


def run(bid_qty_mw: float = DEFAULT_BID_QTY_MW, out_name: str = DEFAULT_OUT_BASENAME) -> dict:
    seeds.seed_everything(42)
    model, stats_obj = _load_f7()
    cal = CalibratedForecaster.from_val_preds(F7_DIR / "val_preds.npz", alpha=0.1)
    df = pl.read_parquet(REPO_ROOT / "data/processed/dk1_panel_val.parquet")
    series = df["imbalance_price_dkk_mwh_15min"].to_numpy().astype(np.float64)
    seq_len = model.seq_len
    n = min(N_BIDS, series.size - seq_len - 1)

    spec_default = default_p2h_spec()
    spec_pypsa = assetspec_from_pypsa_eursec_dk_network(zone="DK1")

    out = {
        "n_windows": int(n),
        "bid_qty_mw": float(bid_qty_mw),
        "tau_grid_eur": list(TAU_GRID_EUR),
        "policies": list(POLICIES),
        "default_spec": _stress_one_spec(spec_default, model, stats_obj, cal, series, seq_len, n, bid_qty_mw),
        "pypsa_spec": _stress_one_spec(spec_pypsa, model, stats_obj, cal, series, seq_len, n, bid_qty_mw),
    }
    # Compute per-cell deltas.
    deltas = []
    for d_cell, p_cell in zip(out["default_spec"]["grid"], out["pypsa_spec"]["grid"]):
        deltas.append(
            {
                "policy": d_cell["policy"],
                "tau_eur": d_cell["tau_eur"],
                "delta_accept_rate": p_cell["accept_rate"] - d_cell["accept_rate"],
                "delta_pi_min_median": (
                    (p_cell["pi_min_median"] or 0.0) - (d_cell["pi_min_median"] or 0.0)
                ),
            }
        )
    out["deltas"] = deltas
    out["summary"] = {
        "default_min_accept": min(c["accept_rate"] for c in out["default_spec"]["grid"]),
        "default_max_accept": max(c["accept_rate"] for c in out["default_spec"]["grid"]),
        "pypsa_min_accept": min(c["accept_rate"] for c in out["pypsa_spec"]["grid"]),
        "pypsa_max_accept": max(c["accept_rate"] for c in out["pypsa_spec"]["grid"]),
        "max_abs_delta_accept": max(abs(d["delta_accept_rate"]) for d in deltas),
    }
    out_path = REPO_ROOT / f"experiments/outputs/{out_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bid-qty", type=float, default=DEFAULT_BID_QTY_MW)
    p.add_argument("--out-name", type=str, default=DEFAULT_OUT_BASENAME)
    args = p.parse_args()
    res = run(bid_qty_mw=args.bid_qty, out_name=args.out_name)
    print(json.dumps(res["summary"], indent=2))
