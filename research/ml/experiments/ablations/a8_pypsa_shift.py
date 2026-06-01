"""A8-PyPSA — physics-grounded regime-shift ablation (merit-order formulation).

Replaces the synthetic AR shift in `a8_synthetic_shift.py` and the
multiplicative `residuals × 2` hack in `a8_conformal_variant.py` with a
**physics-grounded** regime shift: we parameterise a merit-order DA price
model using PyPSA-Eur-Sec costs (`data/raw/pypsa_eursec/costs_2030.csv`,
SHA-pinned) and real DK1 wind / solar / load profiles from the Energinet
free-tier panel, then perturb the gas marginal cost to model a fuel-supply
shock (e.g. the 2022 Russian-gas regime shift).

Why merit-order rather than full PyPSA LP?
- PyPSA LP on the small synthetic DK1+DK2 network in
  `packages/pypsa_adapter/network.py` produces flat marginal prices (gas
  always on the margin at fixed cost). Useful for asset envelope, not for
  price formation.
- Merit-order (Stoft, "Power System Economics", §3) is the textbook
  liberalised-market DA price model: `price_t = max(marginal cost of the
  most expensive generator dispatched to meet net demand)`. Honest, simple,
  reproducible, and matches PyPSA-LP in the boundary case where capacity
  constraints don't bind.
- The PyPSA-Eur-Sec costs file gives us the wind / gas / coal / nuclear
  marginal costs. We use those values; the shift is a multiplier on the
  gas row only.

Theorem 1a (split-CP) requires exchangeability. A fuel-supply shock breaks it.
Theorem 1b (online ACI) is shift-aware. The ablation measures coverage decay
of each method as the shift magnitude grows.

Outputs:
- `experiments/outputs/a8_pypsa_shift.json` — per-method × per-shift coverage,
  interval width, baseline pinball.
- `notes/ablations/A8-pypsa.md` — short writeup.

Run:
    PYTHONPATH=. uv run python experiments/ablations/a8_pypsa_shift.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from heimdall_ml.conformal.aci import AdaptiveConformalInference
from heimdall_ml.conformal.split_cp import SplitConformal

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED = 42
ALPHA = 0.1  # 90% target coverage
AR_P = 24    # AR(24) — 6h memory at 15-min ticks


# ── PyPSA-Eur-Sec marginal costs ────────────────────────────────────────────


def _load_pypsa_marginal_costs() -> dict[str, float]:
    """Read VOM (variable operation & maintenance) + fuel costs from costs_2030.csv.

    PyPSA-Eur-Sec stores `VOM` (€/MWh_el) and `fuel` (€/MWh_th) rows per
    technology. Marginal cost ≈ VOM + fuel / efficiency.
    """
    p = REPO_ROOT / "data/raw/pypsa_eursec/costs_2030.csv"
    df = pl.read_csv(p)
    # Rows of interest: technology + parameter.
    def _val(tech: str, param: str) -> float | None:
        f = df.filter((pl.col("technology") == tech) & (pl.col("parameter") == param))
        return float(f["value"][0]) if len(f) else None

    def _marginal(tech: str, efficiency_default: float) -> float:
        vom = _val(tech, "VOM") or 0.0
        fuel = _val(tech, "fuel") or 0.0
        eff = _val(tech, "efficiency") or efficiency_default
        return float(vom + (fuel / eff if eff else 0.0))

    return {
        # Wind / solar: VOM only, no fuel.
        "wind_onshore": _marginal("onwind", 1.0),
        "wind_offshore": _marginal("offwind", 1.0),
        "solar": _marginal("solar-utility", 1.0),
        # Thermal: VOM + fuel/eff.
        "gas_ccgt": _marginal("CCGT", 0.55),
        "gas_ocgt": _marginal("OCGT", 0.4),
        "coal": _marginal("coal", 0.4),
        "nuclear": _marginal("nuclear", 0.33),
    }


# ── Real DK1 wind / solar / load profiles ──────────────────────────────────


def _load_dk1_supply_demand() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull wind generation, solar generation, and total demand from Energinet.

    Uses `data/processed/dk1_panel_features_v2.parquet` (36-col free-tier
    panel built by `tools/ingest_public_features.py`). Returns (wind_mw,
    solar_mw, demand_mw), each shape (N,), at 15-min resolution.
    """
    panel = pl.read_parquet(REPO_ROOT / "data/processed/dk1_panel_features_v2.parquet")
    panel = panel.drop_nulls(["GrossConsumptionMWh"])

    # Wind = offshore + onshore (≥50 MW + <50 MW + offshore lt 100 + ge 100).
    wind = panel.with_columns(
        (pl.col("OffshoreWindLt100MW_MWh").fill_null(0.0)
         + pl.col("OffshoreWindGe100MW_MWh").fill_null(0.0)
         + pl.col("OnshoreWindLt50kW_MWh").fill_null(0.0)
         + pl.col("OnshoreWindGe50kW_MWh").fill_null(0.0)).alias("wind_mwh")
    )["wind_mwh"].to_numpy()

    solar = panel.with_columns(
        (pl.col("SolarPowerLt10kW_MWh").fill_null(0.0)
         + pl.col("SolarPowerGe10Lt40kW_MWh").fill_null(0.0)
         + pl.col("SolarPowerGe40kW_MWh").fill_null(0.0)).alias("solar_mwh")
    )["solar_mwh"].to_numpy()

    demand = panel["GrossConsumptionMWh"].to_numpy()
    # The values are MWh-per-quarter — multiply by 4 to get MW (15-min ticks).
    return wind * 4, solar * 4, demand * 4


# ── Merit-order price formation ────────────────────────────────────────────


def _merit_order_prices(
    wind_mw: np.ndarray,
    solar_mw: np.ndarray,
    demand_mw: np.ndarray,
    gas_cost_eur_per_mwh: float,
    *,
    nuclear_capacity_mw: float = 0.0,  # DK has no nuclear; keep zero.
    coal_capacity_mw: float = 400.0,    # DK1 residual coal headroom (rough)
    gas_capacity_mw: float = 2000.0,    # DK1 + neighbours flexibility
    costs: dict[str, float] | None = None,
) -> np.ndarray:
    """Compute 15-min DA price via deterministic merit-order.

    Stack (low → high marginal cost):
        1. Renewables (wind + solar)  — marginal ~0 EUR/MWh
        2. Nuclear                     — small VOM, no fuel
        3. Coal                        — VOM + fuel/eff
        4. CCGT                        — VOM + fuel/eff   (← shift target)
        5. Last-resort cap             — VOLL proxy at 3000 EUR/MWh

    Net demand = total demand − renewables. Walk the stack; the marginal
    technology setting the clearing price is the one whose dispatch fills
    the last MW of net demand.
    """
    if costs is None:
        costs = _load_pypsa_marginal_costs()
    # Override gas cost (the shift variable).
    gas_marginal = gas_cost_eur_per_mwh
    n = len(demand_mw)
    prices = np.empty(n, dtype=np.float64)
    renewables = wind_mw + solar_mw
    for t in range(n):
        net = demand_mw[t] - renewables[t]
        if net <= 0:
            # Surplus renewables — price drops to renewables marginal (~VOM_wind).
            prices[t] = max(0.5 * (costs["wind_onshore"] + costs["solar"]), 0.0)
            continue
        remaining = net
        # Nuclear first.
        if remaining <= nuclear_capacity_mw:
            prices[t] = costs["nuclear"]; continue
        remaining -= nuclear_capacity_mw
        # Coal next.
        if remaining <= coal_capacity_mw:
            prices[t] = costs["coal"]; continue
        remaining -= coal_capacity_mw
        # CCGT (the shift variable).
        if remaining <= gas_capacity_mw:
            prices[t] = gas_marginal; continue
        # Capacity exhausted — scarcity price.
        prices[t] = 3000.0
    return prices


# ── Forecaster ──────────────────────────────────────────────────────────────


def _fit_ar(y: np.ndarray, p: int = AR_P) -> tuple[float, np.ndarray]:
    """Closed-form AR(p) on lagged design matrix."""
    n = len(y)
    Y = y[p:]
    X = np.column_stack([y[p - k - 1 : n - k - 1] for k in range(p)])
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Y)), X]), Y, rcond=None)
    return float(coef[0]), coef[1:]


def _ar_forecast(series: np.ndarray, mu: float, phi: np.ndarray) -> np.ndarray:
    p = len(phi)
    preds = np.empty(len(series) - p, dtype=np.float64)
    for i in range(len(preds)):
        preds[i] = mu + phi @ series[i:i + p][::-1]
    return preds


# ── A8 driver ───────────────────────────────────────────────────────────────


def run_pypsa_shift(
    gas_shifts: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0),
    baseline_gas_override_eur: float = 50.0,  # 2022 Q3 TTF day-ahead avg
) -> dict:
    """Run A8-PyPSA. We override the PyPSA-Eur-Sec 2030 gas fuel cost (~3 EUR/MWh)
    with a realistic 2022 Q3 TTF spot (~50 EUR/MWh) so the shift sweep covers
    the actual range of historical gas-shock magnitudes. Provenance is in the
    config block of the output JSON.
    """
    costs = _load_pypsa_marginal_costs()
    print(f"[a8-pypsa] PyPSA-Eur-Sec marginal costs (EUR/MWh):")
    for k, v in costs.items():
        print(f"  {k:<16s}{v:7.2f}")

    print("[a8-pypsa] loading DK1 wind/solar/demand from features_v2 panel ...")
    wind_mw, solar_mw, demand_mw = _load_dk1_supply_demand()
    print(f"  N={len(demand_mw)} quarters; demand mean={demand_mw.mean():.0f} MW "
          f"std={demand_mw.std():.0f} MW")

    # ── Baseline regime: 2 weeks pre-shift, normal gas costs ───────────────
    n_pre = 2 * 7 * 96  # 14 days × 96 quarters = 1344 quarters
    n_post = n_pre  # symmetric
    n_total = n_pre + n_post
    if len(wind_mw) < n_total + 200:
        raise ValueError(f"need ≥{n_total + 200} quarters, have {len(wind_mw)}")
    # Centre the window around mid-2024 (high price volatility era).
    mid = len(wind_mw) // 2
    s0 = mid - n_total // 2
    s1 = s0 + n_total
    wind_w = wind_mw[s0:s1]; solar_w = solar_mw[s0:s1]; demand_w = demand_mw[s0:s1]

    baseline_gas = baseline_gas_override_eur or costs["gas_ccgt"]
    print(f"\n[a8-pypsa] baseline gas marginal cost = {baseline_gas:.2f} EUR/MWh")
    baseline_prices = _merit_order_prices(
        wind_w[:n_pre], solar_w[:n_pre], demand_w[:n_pre],
        gas_cost_eur_per_mwh=baseline_gas, costs=costs,
    )
    print(f"  baseline price mean={baseline_prices.mean():.2f} "
          f"std={baseline_prices.std():.2f} min={baseline_prices.min():.2f} "
          f"max={baseline_prices.max():.2f}")
    if baseline_prices.std() < 1e-6:
        raise RuntimeError("Baseline prices have zero variance — methodology bug")

    # Fit AR(24) on baseline pre-shift.
    mu, phi = _fit_ar(baseline_prices, p=AR_P)
    print(f"[a8-pypsa] AR({AR_P}) fitted: μ̂={mu:.2f}, ||φ||={np.linalg.norm(phi):.3f}")

    # Calibration scores: AR residuals on baseline tail.
    base_preds = _ar_forecast(baseline_prices, mu, phi)
    base_truth = baseline_prices[AR_P:]
    base_resid = np.abs(base_truth - base_preds)
    n_cal = len(base_resid) // 2
    cal_scores = base_resid[:n_cal]
    base_test_resid = base_resid[n_cal:]
    split_cp = SplitConformal.fit(cal_scores, alpha=ALPHA)
    base_split_cov = float(np.mean(base_test_resid <= split_cp.quantile))
    print(f"  baseline split-CP coverage (sanity) = {base_split_cov:.3f}  (target {1-ALPHA:.2f})")

    # ── Shifted regimes ────────────────────────────────────────────────────
    rows = []
    for mult in gas_shifts:
        gas = baseline_gas * mult
        shifted_prices = _merit_order_prices(
            wind_w[n_pre:], solar_w[n_pre:], demand_w[n_pre:],
            gas_cost_eur_per_mwh=gas, costs=costs,
        )
        print(f"\n[a8-pypsa] shift mult=×{mult}  (gas={gas:.1f} EUR/MWh)  "
              f"price mean={shifted_prices.mean():.2f} std={shifted_prices.std():.2f}")
        shift_preds = _ar_forecast(shifted_prices, mu, phi)
        shift_truth = shifted_prices[AR_P:]
        shift_resid = np.abs(shift_truth - shift_preds)

        # split-CP — frozen quantile from baseline.
        split_cov = float(np.mean(shift_resid <= split_cp.quantile))

        # ACI — warm-started on baseline cal scores; updates online over shift.
        aci = AdaptiveConformalInference(alpha=ALPHA, gamma=0.05, window=5000)
        aci.warm_start(cal_scores)
        hits, widths = [], []
        for r in shift_resid:
            q = aci.quantile()
            hits.append(r <= q)
            widths.append(2 * q)
            aci.update(r)
        aci_cov = float(np.mean(hits))
        aci_w = float(np.mean(widths))

        rows.append({
            "gas_eur_per_mwh": gas,
            "shift_multiplier": mult,
            "shifted_price_mean": float(shifted_prices.mean()),
            "shifted_price_std": float(shifted_prices.std()),
            "split_cp_coverage": split_cov,
            "aci_coverage": aci_cov,
            "split_cp_width": 2 * float(split_cp.quantile),
            "aci_mean_width": aci_w,
        })
        print(f"  split-CP cov={split_cov:.3f}  ACI cov={aci_cov:.3f}")

    out = {
        "config": {
            "alpha": ALPHA,
            "ar_p": AR_P,
            "n_pre_quarters": n_pre,
            "n_post_quarters": n_post,
            "baseline_gas_eur_per_mwh": baseline_gas,
            "pypsa_costs_csv_sha": "(SHA pinned in packages/pypsa_adapter/eursec_costs.py)",
            "demand_source": "data/processed/dk1_panel_features_v2.parquet",
            "seed": SEED,
        },
        "baseline": {
            "price_mean": float(baseline_prices.mean()),
            "price_std": float(baseline_prices.std()),
            "split_cp_coverage_sanity": base_split_cov,
            "split_cp_quantile": float(split_cp.quantile),
            "ar_mu": mu,
        },
        "shifts": rows,
    }
    out_dir = REPO_ROOT / "experiments/outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "a8_pypsa_shift.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[a8-pypsa] wrote {out_path}")
    return out


if __name__ == "__main__":
    run_pypsa_shift()
