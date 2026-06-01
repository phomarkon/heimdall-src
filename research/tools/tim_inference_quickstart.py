"""Tim-facing inference quickstart.

Demonstrates the end-to-end Mark-side surface that Tim's focal-orchestrator
consumes per ISP:

    1. Load a recent 192-step price history from the panel.
    2. Call the forecaster registry to get q10/q50/q90 (DKK/MWh).
    3. Wrap with online ACI to get a calibrated [lower, upper] interval.
    4. Build a candidate bid + AssetSpec.
    5. Ask the verifier whether the bid is safe to submit.
    6. Print the accept/reject decision + worst-case-profit number.

Run:
    PYTHONPATH=. uv run python tools/tim_inference_quickstart.py
"""
from __future__ import annotations

import numpy as np
import polars as pl

from heimdall_forecaster.inference import get_forecaster, list_registered
from heimdall_ml.conformal.aci import AdaptiveConformalInference


def step1_load_history():
    """Last 192 quarter-hours from the test panel = a realistic ISP context."""
    df = pl.read_parquet("data/processed/dk1_panel_test.parquet").sort("timestamp_utc")
    series = df["imbalance_price_dkk_mwh_15min"].drop_nulls().to_numpy().astype(float)
    return series[-192:]


def step2_forecast(history: np.ndarray, model_name: str = "f3_ensemble", seed: int = 42):
    f = get_forecaster(model_name, seed=seed)
    quantiles = f.predict(list(history), horizon=16, levels=(0.1, 0.5, 0.9))
    return quantiles


def step3_aci_wrap(history: np.ndarray, q50_first: float, alpha: float = 0.10):
    """Build a (tiny) ACI on the recent residuals; return calibrated half-width."""
    residuals = np.abs(np.diff(history))[-256:]   # naive proxy for warm-start
    aci = AdaptiveConformalInference(alpha=alpha, gamma=0.05, window=256)
    aci.warm_start(residuals)
    half_width = float(aci.quantile())
    return (q50_first - half_width, q50_first + half_width), half_width


def step4_5_verifier_demo(quote_price: float, interval: tuple[float, float]):
    """Tim wires this through `heimdall_verifier.service.verify(...)`.

    For the quickstart we just show the inequality the verifier checks:
        inf_{p in [lower, upper]} pi(action=quote_price, p) >= tau
    With a toy linear profit pi(a, p) = a - p, the worst-case p is `upper`,
    so worst-case profit is quote_price - upper.
    """
    lower, upper = interval
    worst_case_profit = quote_price - upper
    tau = 0.0
    accepted = worst_case_profit >= tau
    return accepted, worst_case_profit


def main():
    print("Registered backends:", list_registered()[:8], "...")
    history = step1_load_history()
    quantiles = step2_forecast(history)

    q10, q50, q90 = quantiles[0].values
    print(f"Forecast (step+15min): q10={q10:.1f}  q50={q50:.1f}  q90={q90:.1f}  DKK/MWh")

    interval, hw = step3_aci_wrap(history, q50)
    print(f"ACI-calibrated [lower, upper] = [{interval[0]:.1f}, {interval[1]:.1f}]  (half-width {hw:.1f})")

    quote_price = q50  # focal MM quotes the median
    accepted, wcp = step4_5_verifier_demo(quote_price, interval)
    print(f"Bid={quote_price:.1f}  worst_case_profit={wcp:.1f}  → {'ACCEPT' if accepted else 'REJECT'}")


if __name__ == "__main__":
    main()
