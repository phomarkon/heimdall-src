"""A8-synthetic — controlled regime-shift ablation. docs/RESEARCH-PROPOSAL.md §5.4.

Replaces the multiplicative residual-doubling hack in `a8_conformal_variant.py`
with a principled synthetic experiment:

1. Generate AR(24) price data with known parameters.
2. Fit an AR(24) forecaster on pre-shift data (no confound — knowable forecaster).
3. Inject controlled variance/mean shifts at the midpoint.
4. Evaluate split-CP (Theorem 1a), ACI (Theorem 1b), and EnbPI at each shift
   magnitude.

The forecaster is the *same* for all shift levels, so the coverage decay is
entirely attributable to the conformal method, not to forecaster quality.

CPU-only. Runs in < 5 s. Produces `experiments/outputs/a8_synthetic_shift.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from heimdall_ml.conformal.aci import AdaptiveConformalInference
from heimdall_ml.conformal.enbpi import enbpi_intervals
from heimdall_ml.conformal.split_cp import SplitConformal

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED = 42

# ── synthetic data generation ────────────────────────────────────────────────


def _decaying_ar_coeffs(p: int, rho: float = 0.85) -> np.ndarray:
    """Return AR(p) coefficients φ_k = ρ^k / S  where S = Σ ρ^k  (sum-to-one)."""
    if rho == 0.0:
        return np.zeros(p)  # white noise — no autoregressive structure
    raw = rho ** np.arange(1, p + 1)
    return raw / raw.sum()


def generate_ar(
    n: int,
    p: int = 24,
    mu: float = 500.0,
    sigma: float = 200.0,
    phi: np.ndarray | None = None,
    seed: int = SEED,
) -> np.ndarray:
    """Generate AR(p) process with Gaussian innovations.

    y_t = μ + Σ φ_k (y_{t-k} - μ) + ε_t,   ε_t ~ N(0, σ²)
    """
    rng = np.random.default_rng(seed)
    if phi is None:
        phi = _decaying_ar_coeffs(p)
    else:
        p = len(phi)
    burn = 200
    y = np.zeros(n + burn)
    y[:p] = mu + rng.normal(0, sigma, size=p)
    for t in range(p, len(y)):
        ar_part = np.dot(phi, y[t - p : t][::-1] - mu)
        y[t] = mu + ar_part + rng.normal(0, sigma)
    return y[burn:]


def _conditional_mean(y: np.ndarray, phi: np.ndarray, mu: float) -> np.ndarray:
    """AR(p) conditional mean: E[y_t | y_{t-p},...,y_{t-1}]."""
    p = len(phi)
    n = len(y) - p
    pred = np.full(len(y), np.nan)
    for t in range(p, len(y)):
        pred[t] = mu + np.dot(phi, y[t - p : t][::-1] - mu)
    return pred[p:]  # first p steps have no prediction


# ── regime shift injection ───────────────────────────────────────────────────


def inject_variance_shift(
    y: np.ndarray, split: int, scale: float
) -> np.ndarray:
    """Multiply post-split values by `scale` centred at pre-split mean.

    Preserves the pre-split mean while scaling deviations around it.  For
    scale > 1 the post-split series has higher variance around the same mean.
    """
    y_shifted = y.copy()
    mu = float(y[:split].mean())
    y_shifted[split:] = y[split:] * scale + mu * (1 - scale)
    return y_shifted


def inject_mean_shift(
    y: np.ndarray, split: int, delta: float
) -> np.ndarray:
    """Add a constant `delta` to every post-split value.

    This shifts the level of the series without changing its variance or
    autocorrelation structure.  For electricity prices this models a step
    change in the market clearing level (e.g. a structural price floor).
    """
    y_shifted = y.copy()
    y_shifted[split:] += delta
    return y_shifted


def inject_structural_break(
    y: np.ndarray,
    split: int,
    phi_old: np.ndarray,
    phi_new: np.ndarray,
    mu: float,
    sigma: float,
    seed: int = SEED,
) -> np.ndarray:
    """Resimulate the post-split segment with new AR coefficients φ_new.

    The pre-split data follows φ_old; post-split follows φ_new.  This
    models a structural change in price dynamics (e.g. new market rules
    changing the autocorrelation structure).
    """
    rng = np.random.default_rng(seed)
    p = len(phi_new)
    y_new = y.copy()
    for t in range(split, len(y)):
        ar_part = np.dot(phi_new, y_new[t - p : t][::-1] - mu)
        y_new[t] = mu + ar_part + rng.normal(0, sigma)
    return y_new


def inject_spikes(
    y: np.ndarray,
    split: int,
    rate: float,
    magnitude: float = 500.0,
    sigma: float = 200.0,
    seed: int = SEED + 1,
) -> np.ndarray:
    """Add Poisson(rate) spikes of random sign × magnitude to post-split data.

    rate = 0.0 → no spikes.  rate = 0.05 → ~5% of post-split steps get a
    spike.  This models the heavy-tailed nature of real electricity prices.
    """
    rng = np.random.default_rng(seed)
    y_spiked = y.copy()
    n_post = len(y) - split
    n_spikes = rng.poisson(rate * n_post)
    if n_spikes == 0:
        return y_spiked
    idxs = rng.choice(n_post, size=min(n_spikes, n_post), replace=False)
    signs = rng.choice([-1.0, 1.0], size=len(idxs))
    for i, s in zip(idxs, signs):
        y_spiked[split + i] += s * magnitude * (1.0 + rng.normal(0, sigma / magnitude))
    return y_spiked


def residuals(y: np.ndarray, phi: np.ndarray, mu: float) -> np.ndarray:
    """Absolute residuals |y_t - E[y_t | past]| — the nonconformity scores."""
    pred = _conditional_mean(y, phi, mu)
    tgt = y[len(phi) :]  # align
    return np.abs(tgt - pred)


# ── conformal evaluation ─────────────────────────────────────────────────────


def _split_cp(scores: np.ndarray) -> dict:
    half = scores.size // 2
    cal, test = scores[:half], scores[half:]
    cp = SplitConformal.fit(cal, alpha=0.1)
    cov = float(np.mean(test <= cp.quantile))
    return {"coverage": cov, "mean_width": 2 * cp.quantile, "n_test": int(test.size)}


def _aci(scores: np.ndarray) -> dict:
    aci = AdaptiveConformalInference(alpha=0.1, gamma=0.05)
    warm = 200
    aci.warm_start(scores[:warm])
    covs, widths = 0, []
    for s in scores[warm:]:
        q = aci.quantile()
        if np.isfinite(q):
            widths.append(2 * q)
        if aci.predict_in_band(float(s)):
            covs += 1
        aci.update(float(s))
    n = len(scores) - warm
    return {
        "coverage": covs / n,
        "mean_width": float(np.mean(widths)) if widths else float("nan"),
        "n_test": n,
    }


def _enbpi(scores: np.ndarray, *, window: int = 200) -> dict:
    """EnbPI with y_t = score_t, point_pred=0.  With symmetric scores this
    reproduces the correct interval width (2× quantile)."""
    targets = scores  # scores are already |y - mu|
    point_pred = np.zeros_like(scores)
    res = enbpi_intervals(
        point_pred=point_pred, targets=targets, alpha=0.1, window=window
    )
    return {
        "coverage": float(res.empirical_coverage),
        "mean_width": float(res.mean_width),
        "n_test": int(res.n_steps),
        "window": window,
    }


def evaluate(scores: np.ndarray) -> dict[str, dict]:
    return {
        "split_cp": _split_cp(scores),
        "aci": _aci(scores),
        "enbpi": _enbpi(scores),
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    n_steps = 5_000
    split = n_steps // 2
    rho = 0.85
    phi = _decaying_ar_coeffs(24, rho)
    mu = 500.0
    sigma = 200.0

    base = generate_ar(n_steps, p=24, mu=mu, sigma=sigma, phi=phi, seed=SEED)
    pre_mu = float(base[:split].mean())

    # ── variance shift sweep ────────────────────────────────────────────────
    var_scales = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    # ── mean shift sweep ────────────────────────────────────────────────────
    mean_deltas = [0.0, 100.0, 200.0, 400.0, 800.0, 1600.0]

    # ── structural break sweep (vary new rho from old) ──────────────────────
    # rho_old = 0.85.  rho_new ∈ {0.85, 0.70, 0.50, 0.30, 0.10, 0.00}
    struct_rhos = [0.85, 0.70, 0.50, 0.30, 0.10, 0.00]

    # ── spike frequency sweep ───────────────────────────────────────────────
    spike_rates = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]

    results: dict = {"params": {
        "n_steps": n_steps, "split_at": split,
        "ar_p": 24, "mu": mu, "sigma": sigma, "base_seed": SEED,
    }}

    def _print_table(title: str, labels: list[str], rows: list[dict], key: str):
        results[key] = {}
        print(f"\n=== {title} ===")
        hdr = f"{'param':>6s}  {'split-CP':>8s} {'width':>8s}  {'ACI':>8s} {'width':>8s}  {'EnbPI':>8s} {'width':>8s}"
        line = "-" * len(hdr)
        print(line)
        print(hdr)
        print(line)
        for label, cell in zip(labels, rows):
            results[key][label] = cell
            print(
                f"{label:>6s}  "
                f"{cell['split_cp']['coverage']:>8.3f} {cell['split_cp']['mean_width']:>8.0f}  "
                f"{cell['aci']['coverage']:>8.3f} {cell['aci']['mean_width']:>8.0f}  "
                f"{cell['enbpi']['coverage']:>8.3f} {cell['enbpi']['mean_width']:>8.0f}"
            )
        print(line)

    # ── variance ──
    _print_table(
        "Variance shift (σ × scale)",
        [f"{s:.1f}" for s in var_scales],
        [evaluate(residuals(base if s == 1.0 else inject_variance_shift(base, split, s), phi, pre_mu)) for s in var_scales],
        "variance_shift",
    )

    # ── mean ──
    _print_table(
        "Mean shift (μ + δ)",
        [f"{d:.0f}" for d in mean_deltas],
        [evaluate(residuals(base if d == 0.0 else inject_mean_shift(base, split, d), phi, pre_mu)) for d in mean_deltas],
        "mean_shift",
    )

    # ── structural ──
    struct_labels = [f"ρ={r:.2f}" for r in struct_rhos]
    struct_rows = []
    for r_new in struct_rhos:
        if r_new == rho:
            y = base
        else:
            phi_new = _decaying_ar_coeffs(24, r_new)
            y = inject_structural_break(base, split, phi, phi_new, mu, sigma, seed=SEED + 99)
        struct_rows.append(evaluate(residuals(y, phi, pre_mu)))
    _print_table("Structural break (ρ old → ρ new)", struct_labels, struct_rows, "structural_break")

    # ── spikes ──
    spike_labels = [f"λ={r:.2f}" for r in spike_rates]
    spike_rows = []
    # Generate one clean base for spikes (no pre-existing shift in base)
    base_clean = generate_ar(n_steps, p=24, mu=mu, sigma=sigma, phi=phi, seed=SEED + 200)
    for rate in spike_rates:
        y = base_clean if rate == 0.0 else inject_spikes(base_clean, split, rate, magnitude=500.0, sigma=sigma)
        spike_rows.append(evaluate(residuals(y, phi, pre_mu)))
    _print_table("Spike frequency (Poisson rate λ)", spike_labels, spike_rows, "spike_frequency")

    out = REPO_ROOT / "experiments/outputs/a8_synthetic_shift.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results → {out}")

    base_cov = results["variance_shift"]["1.0"]["split_cp"]["coverage"]
    if base_cov < 0.85:
        print(f"\nWARNING: split-CP base coverage {base_cov:.3f} < 0.85 — check data.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
