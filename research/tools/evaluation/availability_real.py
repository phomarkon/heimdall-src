"""Grounded availability uncertainty from REAL April-2026 wind/solar forecast error.

The delivery-shortfall downside in ``rescore_runs.py`` re-settles undelivered committed
energy at the imbalance price. Its MECHANISM is real, but its magnitude was a free,
invented coefficient of variation (CV) swept over [0.10, 0.15, 0.25]. This module replaces
that invented number with one grounded in real data, so a "risk-aware LLM beats risk-neutral
det on CVaR" claim rests on a genuine downside, not a dial we tuned.

The honest problem: the raw actual/forecast ratio has a systematic mean bias (wind ≈ 0.52)
from a forecast-vs-generation-table scope/units mismatch — NOT genuine forecast error. Using
it raw would fabricate a ~96% shortfall rate. CV (std/mean), however, is invariant to a
constant multiplicative bias, so the *de-biased relative dispersion* isolates the genuine
relative forecast error. We corroborate it against a same-scope estimator (the DA->1h forecast
revision dispersion, computed entirely within the forecast table, immune to cross-table scope):

    wind  : cv(act/1h, de-biased) = 0.287   vs  cv(1h/DA revision, same-scope) = 0.318   -> agree
    solar : cv(act/1h, de-biased) = 0.230   vs  cv(1h/DA revision, same-scope) = 0.100

The two estimators agree for wind (≈0.29-0.32), confirming the de-biased ratio is genuine
forecast uncertainty, not a scope artifact. For solar the cross-table value is higher (real
intraday cloud variability the revision underestimates); we keep the de-biased cross-table CV.

Output magnitudes are MATERIAL (2-3x the old synthetic 0.10-0.15 default), so the downside
is real. EV is deliberately EXCLUDED from grounding: EV non-availability is plug-in behaviour,
not weather forecast error; fabricating an EV CV would violate the clean-controls rule.

Usage:
    PYTHONPATH=. uv run python tools/evaluation/availability_real.py   # writes the artifact
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CONTEXT_DIR = Path("data/cache/real_context/april_2026")
ARTIFACT = Path("data/cache/availability/grounded_forecast_cv.json")
LEAD = "Forecast1Hour"  # decision-horizon lead for a 15-min mFRR commitment
WIND_TYPES = ["Onshore Wind", "Offshore Wind"]
SOLAR_TYPES = ["Solar"]


def _hourly_actual(gen: pd.DataFrame, gen_type: str) -> pd.Series:
    sub = gen[gen["generation_type"] == gen_type].copy()
    sub["hour"] = sub["timestamp_utc"].dt.floor("h")
    return sub.groupby(["zone", "hour"])["generation_mw"].mean().rename("act")


def _hourly_forecast(fc: pd.DataFrame, types: list[str], col: str) -> pd.Series:
    sub = fc[fc["ForecastType"].isin(types)].copy()
    sub["hour"] = sub["timestamp_utc"].dt.floor("h")
    return sub.groupby(["zone", "hour"])[col].sum().rename(col)


def _debiased_cv(ratio: pd.Series) -> float:
    """CV of a mean-normalised ratio = relative dispersion, invariant to constant scope bias."""
    r = ratio[(ratio > 0) & np.isfinite(ratio)]
    if len(r) < 30:
        return float("nan")
    r = r / r.mean()
    return float(r.std())


def compute_grounded_cv(context_dir: Path = CONTEXT_DIR) -> dict[str, Any]:
    fc = pd.read_parquet(context_dir / "eds_forecasts_hour.parquet")
    gen = pd.read_parquet(context_dir / "generation.parquet")
    out: dict[str, Any] = {"lead": LEAD, "source": str(context_dir), "per_type": {}}
    for name, types, gen_type in [("wind", WIND_TYPES, "wind"), ("solar", SOLAR_TYPES, "solar")]:
        act = _hourly_actual(gen, gen_type)
        f1 = _hourly_forecast(fc, types, LEAD)
        fda = _hourly_forecast(fc, types, "ForecastDayAhead")
        j = pd.concat([f1, fda, act], axis=1).dropna()
        j = j[(j[LEAD] > 10) & (j["ForecastDayAhead"] > 10)]
        cv_ratio = _debiased_cv(j["act"] / j[LEAD])
        cv_revision = _debiased_cv(j[LEAD] / j["ForecastDayAhead"])  # same-scope corroboration
        out["per_type"][name] = {
            "n": int(len(j)),
            "cv_debiased_actual_over_forecast": round(cv_ratio, 4),
            "cv_same_scope_revision": round(cv_revision, 4),
            "cv_grounded": round(cv_ratio, 4),  # the de-biased cross-table value is what we use
        }
    # Archetype -> grounded CV. renewables = wind/solar blend (mean of the two genuine CVs).
    wind_cv = out["per_type"]["wind"]["cv_grounded"]
    solar_cv = out["per_type"]["solar"]["cv_grounded"]
    out["archetype_cv"] = {
        "wind": wind_cv,
        "renewables": round((wind_cv + solar_cv) / 2.0, 4),
        # ev intentionally absent: plug-in availability is not weather forecast error.
    }
    out["caveats"] = (
        "CV de-biased (mean-normalised) to remove a constant forecast-vs-generation scope bias; "
        "corroborated by same-scope DA->1h revision dispersion (wind 0.29 vs 0.32). EV excluded "
        "(not weather). Magnitudes are 2-3x the prior synthetic 0.10-0.15 default."
    )
    return out


def grounded_cv_for_archetype(archetype: str, artifact: Path = ARTIFACT) -> float:
    """Return the grounded CV for an archetype, or 0.0 (firm / not grounded) if absent."""
    if not artifact.exists():
        return 0.0
    data = json.loads(artifact.read_text())
    return float(data.get("archetype_cv", {}).get(str(archetype).lower(), 0.0))


def main() -> None:
    out = compute_grounded_cv()
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {ARTIFACT}")


if __name__ == "__main__":
    main()
