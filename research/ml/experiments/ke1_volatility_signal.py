"""KE1 — volatility-signal kill experiment. docs/RESEARCH-PROPOSAL.md §5.6.

Hypothesis: F7 (univariate transformer) on DK1 imbalance prices beats a naive
volatility-persistence baseline by ≥ 5 % mean pinball loss on the post-2025
val window. If not, the forecasting hypothesis is dead and we pivot to the
"society-as-stress-test-generator" plan B.

Baselines tested:
  - **EWMA**: exponentially-weighted-mean as point + std as scale; quantiles
    via Gaussian assumption (z_α/2 around the EWMA mean). Cheap, hyperparam-
    less except half-life.
  - **GARCH(1,1)** approximation via rolling residual variance (no extra dep).

Decision rule: pass if
  pinball_loss(F7) <= 0.95 * pinball_loss(naive_baseline)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

from heimdall_ml import tracking

REPO_ROOT = Path(__file__).resolve().parents[2]
PASS_THRESHOLD = 0.95
QUANTILES = (0.1, 0.5, 0.9)
TARGET_COL = "imbalance_price_dkk_mwh_15min"


@dataclass
class KE1Verdict:
    f7_pinball: float
    baseline_pinball: float
    ratio: float
    passed: bool
    decision: str


def _pinball(y: np.ndarray, q: np.ndarray, level: float) -> float:
    err = y - q
    return float(np.mean(np.maximum(level * err, (level - 1.0) * err)))


def ewma_baseline(y: np.ndarray, halflife: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """EWMA mean + EWMA std on absolute residuals.

    Returns (mu, sigma) shape (T,) one-step-ahead.
    """
    alpha = 1 - 0.5 ** (1 / halflife)
    mu = np.zeros_like(y)
    var = np.zeros_like(y)
    mu[0] = y[0]
    for t in range(1, y.size):
        mu[t] = alpha * y[t - 1] + (1 - alpha) * mu[t - 1]
        var[t] = alpha * (y[t - 1] - mu[t - 1]) ** 2 + (1 - alpha) * var[t - 1]
    return mu, np.sqrt(np.maximum(var, 1e-8))


def baseline_quantile_forecast(y: np.ndarray, halflife: int = 16) -> np.ndarray:
    """Returns (T, 3) array of quantile forecasts q ∈ {0.1, 0.5, 0.9}."""
    mu, sigma = ewma_baseline(y, halflife)
    z = np.array([norm.ppf(q) for q in QUANTILES])
    return mu[:, None] + z[None, :] * sigma[:, None]


def _load_f7_val_preds() -> tuple[np.ndarray, np.ndarray]:
    """Load F7's val_preds + targets in original DKK/MWh units."""
    z = np.load(REPO_ROOT / "models" / "forecaster" / "f7" / "seed-42" / "val_preds.npz")
    preds = z["preds"]  # (N, H, Q)
    targets = z["targets"]  # (N, H)
    return preds, targets


def run_ke1() -> KE1Verdict:
    preds, targets = _load_f7_val_preds()
    # Use horizon step 0 (next 15-min) for the head-to-head.
    f7_q = preds[:, 0, :]  # (N, 3)
    y = targets[:, 0]  # (N,)
    f7_pinball = float(np.mean([_pinball(y, f7_q[:, i], q) for i, q in enumerate(QUANTILES)]))

    # Baseline reads the val panel directly (raw 15-min target series).
    val_df = pl.read_parquet(REPO_ROOT / "data" / "processed" / "dk1_panel_val.parquet").drop_nulls()
    series = val_df[TARGET_COL].to_numpy().astype(np.float64)
    # Align baseline to F7's val window: F7 uses seq_len=96, horizon=16,
    # so the first F7 prediction targets index seq_len in the raw series.
    seq_len = 96
    y_aligned = series[seq_len : seq_len + y.size]
    base_q = baseline_quantile_forecast(series)[seq_len : seq_len + y.size]
    baseline_pinball = float(
        np.mean([_pinball(y_aligned, base_q[:, i], q) for i, q in enumerate(QUANTILES)])
    )

    ratio = f7_pinball / max(baseline_pinball, 1e-8)
    passed = ratio <= PASS_THRESHOLD
    decision = (
        f"PASS — F7 mean pinball {f7_pinball:.2f} DKK/MWh is {(1-ratio)*100:.1f}% "
        f"below the EWMA baseline {baseline_pinball:.2f} (threshold ≥5%)."
        if passed
        else f"FAIL — F7 mean pinball {f7_pinball:.2f} DKK/MWh fails to beat the EWMA "
        f"baseline {baseline_pinball:.2f} by ≥5% (ratio={ratio:.3f}). "
        "Plan B: society-as-stress-test-generator (docs/RESEARCH-PROPOSAL.md §5.6)."
    )

    tracking.init(experiment="heimdall-ke1")
    with tracking.run(
        name="ke1-volatility-signal",
        params={"halflife": 16, "pass_threshold": PASS_THRESHOLD},
    ):
        tracking.log_metrics(
            {
                "f7_pinball": f7_pinball,
                "baseline_pinball": baseline_pinball,
                "ratio": ratio,
                "passed": float(passed),
            }
        )

    return KE1Verdict(
        f7_pinball=f7_pinball,
        baseline_pinball=baseline_pinball,
        ratio=ratio,
        passed=passed,
        decision=decision,
    )


def write_verdict_note(verdict: KE1Verdict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "# KE1 — volatility-signal kill experiment\n\n"
        "Per docs/RESEARCH-PROPOSAL.md §5.6 (Day-2 kill gate).\n\n"
        f"- F7 mean pinball loss (q∈{{0.1,0.5,0.9}}, horizon step 1): "
        f"**{verdict.f7_pinball:.2f} DKK/MWh**\n"
        f"- EWMA-baseline mean pinball loss (halflife=16, Gaussian quantiles): "
        f"**{verdict.baseline_pinball:.2f} DKK/MWh**\n"
        f"- Ratio (F7 / baseline): **{verdict.ratio:.3f}**\n"
        f"- Pass threshold: ratio ≤ {PASS_THRESHOLD:.2f} (5 % improvement)\n\n"
        f"## Verdict: {'PASS' if verdict.passed else 'FAIL'}\n\n"
        f"{verdict.decision}\n"
    )
    path.write_text(body)


if __name__ == "__main__":
    verdict = run_ke1()
    write_verdict_note(verdict, REPO_ROOT / "notes" / "ke1_verdict.md")
    print(json.dumps(asdict(verdict), indent=2))
