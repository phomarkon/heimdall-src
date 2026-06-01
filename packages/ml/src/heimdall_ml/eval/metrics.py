"""Comprehensive evaluation metrics for probabilistic forecasts.

Single Responsibility: convert (preds, targets, [timestamps]) → a dict of all
metrics a top-tier probabilistic-forecasting reviewer would expect.

Conventions
-----------
- preds:       np.ndarray shape (N, H, Q) — Q quantile predictions at horizons H
- targets:     np.ndarray shape (N, H)
- levels:      tuple of float in (0,1) — the quantile levels predicted, length Q,
               sorted ascending
- timestamps:  np.ndarray of np.datetime64 shape (N,) — issue time of each window
               (optional; required for stratified / temporal metrics)

All metrics are *unitless ratios* OR carry their unit in the metric name suffix
(`_dkk`, `_pct`, `_mwh`). All metrics are pure functions; no IO.

Naming
------
- `*_score` — strictly proper scoring rule (lower better)
- `*_loss`  — non-proper loss (lower better)
- `*_coverage` — empirical coverage (target-dependent; sharpness must accompany)
- `*_width` — interval width / sharpness (lower better at fixed coverage)
- `*_skill` — skill vs reference (positive = better; 0 = same as ref)

References
----------
- Gneiting & Raftery 2007 — proper scoring rules
- Gneiting & Ranjan 2011 — twCRPS (tail-weighted CRPS)
- Romano et al 2019 — CQR
- Gibbs & Candès 2021 — ACI
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


# ---------- 1. Core scoring rules --------------------------------------------------

def pinball_per_quantile(
    preds: np.ndarray, targets: np.ndarray, levels: tuple[float, ...]
) -> dict[str, float]:
    """Mean pinball loss at each quantile level.

    pinball(y, q, l) = max(l * (y - q), (l - 1) * (y - q))
    """
    assert preds.shape[:2] == targets.shape, (preds.shape, targets.shape)
    assert preds.shape[2] == len(levels)
    out: dict[str, float] = {}
    for qi, l in enumerate(levels):
        err = targets - preds[..., qi]
        out[f"pinball_q{int(round(l * 100)):02d}"] = float(
            np.mean(np.maximum(l * err, (l - 1) * err))
        )
    out["pinball_mean"] = float(np.mean(list(out.values())))
    return out


def crps_from_quantiles(
    preds: np.ndarray, targets: np.ndarray, levels: tuple[float, ...]
) -> float:
    """CRPS approximation from Q discrete quantile predictions.

    CRPS = 2 * E_l [ pinball(y, q_l, l) ] for l ~ Uniform(0,1).
    With finite quantile levels, approximated by the trapezoidal rule:

        CRPS ≈ 2 * sum_i w_i * pinball(y, q_i, l_i)

    where w_i are trapezoidal weights on the level grid {l_1,...,l_Q}.
    This is the standard pinball→CRPS reduction (Gneiting & Raftery 2007 §6).
    Sound estimator for the quantile-regression setting; exact only as Q → ∞.
    """
    assert preds.shape[:2] == targets.shape
    L = np.asarray(levels, dtype=np.float64)
    assert L.ndim == 1 and len(L) == preds.shape[2]
    assert np.all(np.diff(L) > 0), "levels must be strictly ascending"
    # Trapezoidal weights on [L_0, L_-1]; we treat the predicted quantiles as a
    # discretisation of the inverse CDF over the level interval covered.
    # Pinball-to-CRPS: integrate 2*pinball over l in (0,1). Outside [L_0,L_-1]
    # we have no information; report the CRPS approximation restricted to
    # the covered band (under-estimates true CRPS — flag clearly).
    err = targets[..., None] - preds  # (N,H,Q)
    pinball = np.maximum(L * err, (L - 1) * err)  # (N,H,Q)
    pinball_per_l = pinball.mean(axis=(0, 1))  # (Q,)
    # Trapezoidal integral approximation: 2 * ∫_{L_0}^{L_-1} pinball(l) dl
    # ≈ 2 * sum (L_{i+1} - L_{i-1}) / 2 * pinball_i   (endpoint half-weights)
    if len(L) == 1:
        return float(2.0 * pinball_per_l[0])
    w = np.zeros_like(L)
    w[0] = (L[1] - L[0]) / 2
    w[-1] = (L[-1] - L[-2]) / 2
    if len(L) > 2:
        w[1:-1] = (L[2:] - L[:-2]) / 2
    return float(2.0 * np.sum(w * pinball_per_l))


def twcrps_tail(
    preds: np.ndarray, targets: np.ndarray, levels: tuple[float, ...],
    tail: str = "upper", threshold_quantile: float = 0.9,
) -> float:
    """Tail-weighted CRPS via the threshold-weight function of Gneiting & Ranjan 2011.

    w(z) = 1{z > t}  for upper tail,  1{z < t} for lower tail
    where t is the empirical threshold_quantile of `targets`.

    Reduces CRPS contribution to the tail region only — material for risk-aware
    bidding (the verifier reasons about worst-case profit). Returns mean over
    all windows.
    """
    t = float(np.quantile(targets, threshold_quantile if tail == "upper" else 1 - threshold_quantile))
    if tail == "upper":
        weight_target = (targets > t).astype(np.float64)
    elif tail == "lower":
        weight_target = (targets < t).astype(np.float64)
    else:
        raise ValueError(f"tail must be 'upper' or 'lower'; got {tail!r}")
    if weight_target.sum() == 0:
        return float("nan")
    L = np.asarray(levels, dtype=np.float64)
    err = targets[..., None] - preds
    pinball = np.maximum(L * err, (L - 1) * err)
    # Per-window mean pinball, then restrict to tail windows.
    per_window = pinball.mean(axis=(1, 2))  # (N,)
    # Use the worst-target horizon per window as the tail decision.
    target_per_window = targets.max(axis=1) if tail == "upper" else targets.min(axis=1)
    if tail == "upper":
        mask = target_per_window > t
    else:
        mask = target_per_window < t
    if mask.sum() == 0:
        return float("nan")
    return float(per_window[mask].mean())


# ---------- 2. Calibration --------------------------------------------------------

def marginal_coverage_per_level(
    preds: np.ndarray, targets: np.ndarray, levels: tuple[float, ...],
) -> dict[str, float]:
    """Per-quantile empirical CDF coverage: P(target <= q_l) for each level l.

    A well-calibrated model has P(y <= q_l) ≈ l.
    """
    out: dict[str, float] = {}
    for qi, l in enumerate(levels):
        cov = float(np.mean(targets <= preds[..., qi]))
        out[f"cdf_at_q{int(round(l * 100)):02d}"] = cov
    return out


def interval_coverage(
    preds: np.ndarray, targets: np.ndarray, levels: tuple[float, ...],
    nominal_pairs: tuple[tuple[float, float], ...] = ((0.1, 0.9),),
) -> dict[str, float]:
    """Empirical coverage of [q_lo, q_hi] intervals plus mean width."""
    out: dict[str, float] = {}
    L = np.asarray(levels)
    for lo, hi in nominal_pairs:
        lo_i = int(np.argmin(np.abs(L - lo)))
        hi_i = int(np.argmin(np.abs(L - hi)))
        nom = float(hi - lo)
        lo_p = preds[..., lo_i]
        hi_p = preds[..., hi_i]
        # Robust against q crossing.
        lo_p2 = np.minimum(lo_p, hi_p)
        hi_p2 = np.maximum(lo_p, hi_p)
        cov = float(np.mean((targets >= lo_p2) & (targets <= hi_p2)))
        width = float(np.mean(hi_p2 - lo_p2))
        tag = f"{int(round(lo*100)):02d}_{int(round(hi*100)):02d}"
        out[f"interval_{tag}_coverage"] = cov
        out[f"interval_{tag}_nominal"] = nom
        out[f"interval_{tag}_width_mean"] = width
    return out


def pit_histogram(
    preds: np.ndarray, targets: np.ndarray, levels: tuple[float, ...],
    bins: int = 20,
) -> dict[str, Any]:
    """PIT histogram + uniformity test.

    PIT_i = F̂_i(y_i) ≈ the level l_i ∈ levels for which q_{l_i} is closest to
    target from below. Coarse with few quantiles; report KS statistic vs U(0,1)
    for a single-number summary and the histogram for plotting.
    """
    L = np.asarray(levels, dtype=np.float64)
    # Linearly interpolate within the predicted quantile function to obtain a
    # smooth PIT (the step-function PIT concentrates on |L|+1 distinct values
    # and is structurally non-uniform even when the model is calibrated).
    flat_preds = np.sort(preds.reshape(-1, preds.shape[-1]), axis=-1)
    flat_targets = targets.reshape(-1)
    pit = np.empty(flat_targets.shape[0])
    Q = flat_preds.shape[-1]
    for i in range(flat_targets.shape[0]):
        sorted_p = flat_preds[i]
        y = flat_targets[i]
        if y <= sorted_p[0]:
            # Extrapolate below first quantile using slope to L[0] from L=0.
            if sorted_p[0] > -np.inf:
                # Linear from (q=-inf, L=0) → (sorted_p[0], L[0]) is undefined;
                # clip to L[0] adjusted by distance to next quantile.
                if Q > 1 and sorted_p[1] > sorted_p[0]:
                    slope = (L[1] - L[0]) / (sorted_p[1] - sorted_p[0])
                    pit[i] = max(0.0, L[0] - slope * (sorted_p[0] - y))
                else:
                    pit[i] = 0.0
            else:
                pit[i] = 0.0
        elif y >= sorted_p[-1]:
            if Q > 1 and sorted_p[-1] > sorted_p[-2]:
                slope = (L[-1] - L[-2]) / (sorted_p[-1] - sorted_p[-2])
                pit[i] = min(1.0, L[-1] + slope * (y - sorted_p[-1]))
            else:
                pit[i] = 1.0
        else:
            j = int(np.searchsorted(sorted_p, y, side="right") - 1)
            j = max(0, min(Q - 2, j))
            denom = sorted_p[j + 1] - sorted_p[j]
            frac = 0.5 if denom <= 0 else (y - sorted_p[j]) / denom
            pit[i] = L[j] + frac * (L[j + 1] - L[j])
    hist, edges = np.histogram(pit, bins=bins, range=(0.0, 1.0), density=False)
    # KS vs uniform: empirical CDF at right edge of bin i is cumsum/total;
    # uniform CDF at right edge of bin i (i=0..bins-1) is (i+1)/bins.
    cum = np.cumsum(hist) / hist.sum()
    uniform_cdf = np.arange(1, len(cum) + 1) / len(cum)
    ks = float(np.max(np.abs(cum - uniform_cdf)))
    return {
        "bins": int(bins),
        "edges": edges.tolist(),
        "counts": hist.tolist(),
        "ks_vs_uniform": ks,
    }


def reliability_curve(
    preds: np.ndarray, targets: np.ndarray, levels: tuple[float, ...],
) -> dict[str, list[float]]:
    """Nominal vs empirical coverage at each predicted quantile level.

    For a perfectly calibrated model, empirical == nominal at every level.
    """
    nominal = list(map(float, levels))
    empirical = []
    for qi, l in enumerate(levels):
        empirical.append(float(np.mean(targets <= preds[..., qi])))
    return {"nominal": nominal, "empirical": empirical}


# ---------- 3. Conditional / stratified ------------------------------------------

def metrics_by_stratum(
    preds: np.ndarray, targets: np.ndarray, levels: tuple[float, ...],
    stratum: np.ndarray, stratum_name: str = "stratum",
) -> dict[str, dict[str, float]]:
    """Pinball-mean + 80% interval coverage per unique stratum value."""
    out: dict[str, dict[str, float]] = {}
    uniq = np.unique(stratum)
    for val in uniq:
        mask = stratum == val
        if mask.sum() == 0:
            continue
        sub_preds = preds[mask]
        sub_targets = targets[mask]
        m = pinball_per_quantile(sub_preds, sub_targets, levels)
        m.update(interval_coverage(sub_preds, sub_targets, levels))
        m["n"] = int(mask.sum())
        out[f"{stratum_name}={val}"] = m
    return out


# ---------- 4. Tail / event detection --------------------------------------------

def spike_detection(
    preds: np.ndarray, targets: np.ndarray, levels: tuple[float, ...],
    threshold_quantile: float = 0.95,
) -> dict[str, float]:
    """Binary event = target > empirical threshold_quantile of targets.
    Use median forecast as score.
    """
    L = np.asarray(levels)
    mid_i = int(np.argmin(np.abs(L - 0.5)))
    score = preds[..., mid_i]
    thr = float(np.quantile(targets, threshold_quantile))
    y = (targets > thr).astype(np.float64)
    s = score.flatten()
    y_flat = y.flatten()
    # Brier on flattened predictions: use the *probability that target > thr*
    # from the empirical CDF over quantiles. Cheap proxy:
    p_event = np.mean(preds > thr, axis=-1).flatten()
    brier = float(np.mean((p_event - y_flat) ** 2))
    # AUC: rank by p_event
    order = np.argsort(-p_event)
    y_ord = y_flat[order]
    tp = np.cumsum(y_ord)
    fp = np.cumsum(1 - y_ord)
    if tp[-1] == 0 or fp[-1] == 0:
        auc = float("nan")
    else:
        tpr = tp / tp[-1]
        fpr = fp / fp[-1]
        auc = float(np.trapezoid(tpr, fpr))
    return {
        "spike_threshold_quantile": float(threshold_quantile),
        "spike_threshold_value": thr,
        "spike_brier": brier,
        "spike_auc": auc,
        "spike_base_rate": float(y_flat.mean()),
    }


def expected_shortfall_coverage(
    preds: np.ndarray, targets: np.ndarray, levels: tuple[float, ...],
    alpha: float = 0.95,
) -> dict[str, float]:
    """Empirical coverage of the ES at level alpha using the predicted q_alpha.

    ES_alpha(y) = E[y | y >= VaR_alpha(y)]. We approximate predicted ES from
    available high quantiles by averaging quantile predictions at levels >= alpha.
    Reports (predicted ES - realized ES) per window then aggregates.
    """
    L = np.asarray(levels)
    tail_mask_l = L >= alpha
    if tail_mask_l.sum() == 0:
        return {f"es{int(alpha*100)}_n_quantiles": 0}
    pred_es = preds[..., tail_mask_l].mean(axis=-1).flatten()
    target_flat = targets.flatten()
    realized_es = float(target_flat[target_flat >= np.quantile(target_flat, alpha)].mean())
    pred_es_mean = float(pred_es.mean())
    return {
        f"es{int(alpha*100)}_predicted_mean_dkk": pred_es_mean,
        f"es{int(alpha*100)}_realized_dkk": realized_es,
        f"es{int(alpha*100)}_bias_dkk": pred_es_mean - realized_es,
    }


# ---------- 5. Decision-theoretic ------------------------------------------------

def simple_bid_pnl(
    preds: np.ndarray, targets: np.ndarray, levels: tuple[float, ...],
    bid_quantile: float = 0.5,
    award_rule: str = "all_awarded",
) -> dict[str, float]:
    """P&L of a placeholder bidding policy that submits the q_bid quantile as price.

    `award_rule='all_awarded'` — assumes every bid clears. P&L per quarter-hour
    is (target - bid) for a buy-side actor (focal P2H asset acquires energy
    when imbalance price is favourable). This is *illustrative*; the verifier
    layer is what makes the real-money story; here we report a single scalar
    so reviewers can rank forecasters by downstream economic value.
    """
    L = np.asarray(levels)
    qi = int(np.argmin(np.abs(L - bid_quantile)))
    bid = preds[..., qi]
    pnl = targets - bid
    return {
        "pnl_total_dkk": float(pnl.sum()),
        "pnl_mean_dkk_per_quarter": float(pnl.mean()),
        "pnl_std_dkk": float(pnl.std()),
        "pnl_n_quarters": int(pnl.size),
        "bid_quantile": float(bid_quantile),
    }


# ---------- 6. Skill scores ------------------------------------------------------

def skill_score(model_score: float, reference_score: float) -> float:
    """1 - model/ref. Positive = better than reference."""
    if reference_score == 0 or not np.isfinite(reference_score):
        return float("nan")
    return float(1.0 - model_score / reference_score)


# ---------- 7. Persistence reference ---------------------------------------------

def persistence_forecast(history_targets: np.ndarray, horizon: int) -> np.ndarray:
    """Naive persistence: predicted q50 = last observed target.

    history_targets shape (N, T) — last T observations per window
    returns preds shape (N, horizon, 1) — single quantile (q50 by convention)
    """
    last = history_targets[:, -1:]  # (N,1)
    return np.broadcast_to(last[:, :, None], (history_targets.shape[0], horizon, 1)).copy()


# ---------- 8. Orchestrator ------------------------------------------------------

def collect_all(
    preds: np.ndarray, targets: np.ndarray, levels: tuple[float, ...],
    timestamps: np.ndarray | None = None,
    reference_crps: float | None = None,
) -> dict[str, Any]:
    """Compute the full reviewer-defence metric bundle for one (model, seed)."""
    out: dict[str, Any] = {
        "n_windows": int(preds.shape[0]),
        "n_horizons": int(preds.shape[1]),
        "n_quantiles": int(preds.shape[2]),
        "levels": list(map(float, levels)),
    }

    # 1. Proper scoring rules
    out.update(pinball_per_quantile(preds, targets, levels))
    out["crps_quantile_dkk"] = crps_from_quantiles(preds, targets, levels)
    out["twcrps_upper_q90_dkk"] = twcrps_tail(preds, targets, levels, "upper", 0.9)
    out["twcrps_lower_q10_dkk"] = twcrps_tail(preds, targets, levels, "lower", 0.1)

    # 2. Calibration
    out.update(marginal_coverage_per_level(preds, targets, levels))
    out.update(interval_coverage(preds, targets, levels, ((0.1, 0.9), (0.25, 0.75))))
    out["pit"] = pit_histogram(preds, targets, levels)
    out["reliability"] = reliability_curve(preds, targets, levels)

    # 3. Tail / event
    out.update(spike_detection(preds, targets, levels, threshold_quantile=0.95))
    out.update(expected_shortfall_coverage(preds, targets, levels, alpha=0.95))

    # 4. Decision
    out.update(simple_bid_pnl(preds, targets, levels, bid_quantile=0.5))

    # 5. Skill
    if reference_crps is not None and reference_crps > 0:
        out["crps_skill_vs_reference"] = skill_score(out["crps_quantile_dkk"], reference_crps)

    # 6. Stratified (only if timestamps given)
    if timestamps is not None and len(timestamps) == preds.shape[0]:
        ts = np.asarray(timestamps, dtype="datetime64[h]")
        hour = (ts.astype("int64") % 24).astype(np.int64)
        out["by_hour"] = metrics_by_stratum(preds, targets, levels, hour, "hour_of_day")
        weekday = ((ts.astype("datetime64[D]").astype("int64") + 4) % 7).astype(np.int64)
        out["by_weekday"] = metrics_by_stratum(preds, targets, levels, weekday, "weekday")
        month = (ts.astype("datetime64[M]").astype("int64") % 12 + 1).astype(np.int64)
        out["by_month"] = metrics_by_stratum(preds, targets, levels, month, "month")

    return out


__all__ = [
    "collect_all",
    "crps_from_quantiles",
    "expected_shortfall_coverage",
    "interval_coverage",
    "marginal_coverage_per_level",
    "metrics_by_stratum",
    "persistence_forecast",
    "pinball_per_quantile",
    "pit_histogram",
    "reliability_curve",
    "simple_bid_pnl",
    "skill_score",
    "spike_detection",
    "twcrps_tail",
]
