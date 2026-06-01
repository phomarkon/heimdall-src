"""Classical statistical baselines for the reviewer-proofing track (Plan v2 B).

B5 SARIMAX with exogenous regressors, B6 GARCH(1,1), B8 Prophet, B9 sparse GP.

Each baseline outputs (q10, q50, q90) quantile predictions over the val horizon
and an entry into the leaderboard. Most heavy lifting is in fitting; predict
loops are tiny.

USAGE
    uv run python experiments/baselines_classical.py --baselines b5 b6 b8 b9

Heavy deps (statsmodels in core; arch / prophet / gpytorch are optional):
    uv add arch prophet gpytorch

If an optional dep is missing, the corresponding baseline is skipped with a
logged warning, and the other baselines still run.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "outputs"
TARGET = "imbalance_price_dkk_mwh_15min"

# Per docs/RESEARCH-PROPOSAL.md §5.7
PRE_POST_BREAK_UTC = datetime(2025, 3, 4, 0, 0, tzinfo=UTC)
TEST_START_UTC = datetime(2025, 5, 1, 0, 0, tzinfo=UTC)


def _load_panels() -> tuple[pl.DataFrame, pl.DataFrame]:
    train = pl.read_parquet(PROCESSED / "dk1_panel_rich_train.parquet")
    val = pl.read_parquet(PROCESSED / "dk1_panel_rich_val.parquet")
    return train, val


def _pinball(y: np.ndarray, q: np.ndarray, level: float) -> float:
    err = y - q
    return float(np.mean(np.maximum(level * err, (level - 1.0) * err)))


def _record_metrics(name: str, y_val: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray):
    p10 = _pinball(y_val, q10, 0.1)
    p50 = _pinball(y_val, q50, 0.5)
    p90 = _pinball(y_val, q90, 0.9)
    cov = float(((y_val >= q10) & (y_val <= q90)).mean())
    return {
        "name": name,
        "val_pinball_q10": p10,
        "val_pinball_q50": p50,
        "val_pinball_q90": p90,
        "val_pinball_mean_dkk": (p10 + p50 + p90) / 3.0,
        "val_q10_q90_coverage": cov,
        "n_val": len(y_val),
    }


# ─────────────────────────────── B5 SARIMAX ────────────────────────────────


def fit_b5_sarimax(train: pl.DataFrame, val: pl.DataFrame, exog_cols: tuple[str, ...]) -> dict:
    from scipy.stats import norm
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y_tr = train[TARGET].to_numpy().astype(np.float64)
    y_va = val[TARGET].to_numpy().astype(np.float64)
    X_tr = train.select(list(exog_cols)).to_numpy().astype(np.float64) if exog_cols else None
    X_va = val.select(list(exog_cols)).to_numpy().astype(np.float64) if exog_cols else None

    # Subsample for fit (SARIMAX is O(n²) — full 175k is intractable).
    sub_n = min(40_000, len(y_tr))
    y_fit = y_tr[-sub_n:]
    X_fit = X_tr[-sub_n:] if X_tr is not None else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            y_fit,
            exog=X_fit,
            order=(2, 0, 2),
            seasonal_order=(1, 0, 1, 96),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=50)

    pred = res.get_forecast(steps=len(y_va), exog=X_va)
    mean = pred.predicted_mean
    se = np.sqrt(np.maximum(pred.var_pred_mean, 1e-6))
    z10, z90 = norm.ppf(0.1), norm.ppf(0.9)
    q50 = mean
    q10 = mean + z10 * se
    q90 = mean + z90 * se
    return _record_metrics("b5_sarimax", y_va, q10, q50, q90)


# ─────────────────────────────── B6 GARCH ──────────────────────────────────


def fit_b6_garch(train: pl.DataFrame, val: pl.DataFrame) -> dict:
    try:
        from arch import arch_model
    except ImportError:
        return {"name": "b6_garch", "error": "arch not installed; uv add arch"}
    from scipy.stats import norm

    y_tr = train[TARGET].to_numpy().astype(np.float64)
    y_va = val[TARGET].to_numpy().astype(np.float64)
    # Fit GARCH(1,1) to residuals after subtracting train mean.
    mu = float(y_tr.mean())
    resid = y_tr - mu
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(resid, vol="Garch", p=1, q=1, mean="Constant", rescale=False)
        res = am.fit(disp="off")
    # Forecast: persist mean, propagate conditional volatility.
    fc = res.forecast(horizon=len(y_va), reindex=False)
    var = fc.variance.values[-1]
    se = np.sqrt(np.maximum(var, 1e-6))
    z10, z90 = norm.ppf(0.1), norm.ppf(0.9)
    q50 = np.full(len(y_va), mu)
    q10 = q50 + z10 * se
    q90 = q50 + z90 * se
    return _record_metrics("b6_garch", y_va, q10, q50, q90)


# ─────────────────────────────── B8 Prophet ────────────────────────────────


def fit_b8_prophet(train: pl.DataFrame, val: pl.DataFrame) -> dict:
    try:
        from prophet import Prophet
    except ImportError:
        return {"name": "b8_prophet", "error": "prophet not installed; uv add prophet"}

    df_tr = train.select([
        pl.col("timestamp_utc").alias("ds"),
        pl.col(TARGET).alias("y"),
    ]).to_pandas()
    df_va = val.select([pl.col("timestamp_utc").alias("ds")]).to_pandas()
    df_tr["ds"] = df_tr["ds"].dt.tz_convert(None)
    df_va["ds"] = df_va["ds"].dt.tz_convert(None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(interval_width=0.8, weekly_seasonality=True, daily_seasonality=True)
        m.add_country_holidays(country_name="DK")
        m.fit(df_tr)
        fc = m.predict(df_va)
    return _record_metrics(
        "b8_prophet",
        val[TARGET].to_numpy().astype(np.float64),
        fc["yhat_lower"].to_numpy(),
        fc["yhat"].to_numpy(),
        fc["yhat_upper"].to_numpy(),
    )


# ─────────────────────────────── B9 Sparse GP ──────────────────────────────


def fit_b9_sgp(train: pl.DataFrame, val: pl.DataFrame) -> dict:
    try:
        import gpytorch
        import torch
    except ImportError:
        return {"name": "b9_sgp", "error": "gpytorch not installed; uv add gpytorch"}
    from scipy.stats import norm

    # Subsample for tractability: 10k inducing + 30k training points.
    y_tr_full = train[TARGET].to_numpy().astype(np.float32)
    n_tr = min(30_000, len(y_tr_full))
    idx = np.random.default_rng(13).choice(len(y_tr_full), size=n_tr, replace=False)
    idx.sort()
    y_tr = y_tr_full[idx]
    X_tr = np.arange(len(y_tr_full), dtype=np.float32)[idx][:, None]
    y_va = val[TARGET].to_numpy().astype(np.float32)
    X_va = np.arange(len(y_tr_full), len(y_tr_full) + len(y_va), dtype=np.float32)[:, None]

    n_ind = min(2000, n_tr // 5)
    ind_idx = np.linspace(0, n_tr - 1, n_ind, dtype=int)
    X_ind = X_tr[ind_idx]

    class SVGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing):
            vd = gpytorch.variational.CholeskyVariationalDistribution(inducing.size(0))
            vs = gpytorch.variational.VariationalStrategy(self, inducing, vd, learn_inducing_locations=False)
            super().__init__(vs)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    Xt = torch.tensor(X_tr)
    yt = torch.tensor(y_tr)
    Xind = torch.tensor(X_ind)
    Xv = torch.tensor(X_va)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SVGPModel(Xind)
    optimiser = torch.optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.05)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_tr)
    model.train()
    likelihood.train()
    for _ in range(50):  # quick fit; user can deepen
        optimiser.zero_grad()
        output = model(Xt)
        loss = -mll(output, yt)
        loss.backward()
        optimiser.step()
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        posterior = likelihood(model(Xv))
        mu = posterior.mean.numpy()
        var = posterior.variance.numpy()
    se = np.sqrt(np.maximum(var, 1e-6))
    z10, z90 = norm.ppf(0.1), norm.ppf(0.9)
    return _record_metrics("b9_sgp", y_va.astype(np.float64), mu + z10 * se, mu, mu + z90 * se)


# ─────────────────────────────── runner ────────────────────────────────────


BASELINES: dict[str, Callable[[pl.DataFrame, pl.DataFrame], dict]] = {
    "b5": lambda t, v: fit_b5_sarimax(
        t,
        v,
        exog_cols=(
            "load_actual_mw",
            "da_price_dkk_mwh",
            "wind_speed_100m",
            "shortwave_radiation",
        ),
    ),
    "b6": fit_b6_garch,
    "b8": fit_b8_prophet,
    "b9": fit_b9_sgp,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", nargs="+", default=list(BASELINES.keys()))
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, val = _load_panels()
    results: list[dict] = []
    for name in args.baselines:
        if name not in BASELINES:
            print(f"[skip] unknown baseline: {name}")
            continue
        print(f"[run] {name} ...", flush=True)
        try:
            r = BASELINES[name](train, val)
        except Exception as exc:
            r = {"name": name, "error": f"{type(exc).__name__}: {exc}"}
        print(f"  -> {json.dumps(r, indent=None)}")
        results.append(r)

    out = OUTPUT_DIR / "baselines_classical.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
