"""B3 baseline — Hagström & Herre (2025) re-implementation.

Per docs/RESEARCH-PROPOSAL.md §5.2 (B3) and §5.3.1 head-to-head protocol.

Hagström & Herre (arXiv:2511.19715, 2025) — *Understanding Risk and Revenue
in the Nordic 15-Minute mFRR Market: An EV Aggregation Study* — a
CVaR-augmented two-stage stochastic-optimisation framework for EV-fleet
participation in the post-March-2025 Nordic mFRR EAM.  We re-implement the
algorithmic pipeline as faithfully as the available description allows and
substitute the EV virtual-battery with a **PyPSA-Eur-Sec-derived P2H +
multi-horizon thermal-storage virtual battery** of identical interface
(capacity-energy envelopes computed via the same procedure).

Comparator pipeline (per §5.2):
  Stage 1 — day-ahead bid quantity per 15-min MTU, committed before the day.
  Stage 2 — per-MTU mFRR adjustment, conditional on the realised imbalance
            scenario.  Objective: maximise  E[profit] − λ·CVaR_{1-α}(profit).
  Constraints: virtual-battery SoC envelope, ramp, capacity.

Evaluation window (proposal §5.2): **2025-04-01 → 2025-12-31** (post-break,
identical for both methods).  Metrics: realised €/MWh, Sharpe, CVaR(5%) of
daily P&L, mFRR participation rate.

The implementation here is solver-backed (cvxpy + clarabel/scs) at the day
level — 96 ticks per LP — with a rolling-horizon outer loop.  Scenarios are
drawn from the empirical residual distribution of imbalance vs. day-ahead in
the trailing 7-day window, i.e. exactly the H&H §3.2 "empirical scenario
generator."

References:
- Hagström, A. and Herre, L. (2025). Understanding Risk and Revenue in the
  Nordic 15-Minute mFRR Market: An EV Aggregation Study. arXiv:2511.19715.
- ENTSO-E balancing-market evolution briefing 2025-Q1 (mFRR EAM go-live).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cvxpy as cp
import numpy as np
import polars as pl

from experiments.baselines_focal_agent import (
    DKK_PER_EUR,
    P2HSpec,
    TICKS_PER_DAY,
    ANNUAL_TICKS,
    EvalResult,
    load_panel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class B3Config:
    eval_window_start_utc: str = "2025-04-01T00:00:00Z"
    eval_window_end_utc: str = "2025-12-31T23:45:00Z"
    n_scenarios: int = 24                 # H&H §3.2 — verify
    cvar_alpha: float = 0.05              # 5% tail
    cvar_lambda: float = 0.5              # weight on CVaR penalty
    horizon_ticks: int = TICKS_PER_DAY    # one-day rolling horizon
    residual_window_days: int = 7
    seed: int = 42
    out_dir: Path = field(default_factory=lambda: REPO_ROOT / "experiments/outputs/baselines_b3")


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------


def _imb_scenarios(
    history_da: np.ndarray,
    history_imb: np.ndarray,
    forecast_da: np.ndarray,
    n_scenarios: int,
    rng: np.random.Generator,
    residual_window_ticks: int,
) -> np.ndarray:
    """Draw ``n_scenarios`` of length ``len(forecast_da)`` paths for imbalance.

    Resamples residuals (imb − da) from the trailing window with replacement —
    a non-parametric block-bootstrap-lite that matches the H&H "empirical
    scenario generator." Returns an (S, H) array of imbalance prices.
    """
    H = len(forecast_da)
    if history_da.size < 8:
        return np.broadcast_to(forecast_da, (n_scenarios, H)).copy()
    w_lo = max(0, history_da.size - residual_window_ticks)
    res = history_imb[w_lo:] - history_da[w_lo:]
    if res.size == 0:
        return np.broadcast_to(forecast_da, (n_scenarios, H)).copy()
    draws = rng.choice(res, size=(n_scenarios, H), replace=True)
    return forecast_da[None, :] + draws


# ---------------------------------------------------------------------------
# Two-stage stochastic LP per rolling horizon
# ---------------------------------------------------------------------------


def _solve_horizon_lp(
    da_path: np.ndarray,        # (H,)
    scen_imb: np.ndarray,        # (S, H)
    soc_init: float,
    spec: P2HSpec,
    cfg: B3Config,
) -> np.ndarray:
    """Solve the two-stage LP and return the *committed* per-tick action q_t MW
    (the first-stage variable; H entries).  Sign convention matches
    ``baselines_focal_agent.evaluate_policy`` — q>0 ⇒ down (consume), q<0 ⇒ up
    (curtail)."""
    S, H = scen_imb.shape
    cap = spec.capacity_mw

    q = cp.Variable(H)                       # first-stage commitment per tick
    a = cp.Variable((S, H))                  # second-stage mFRR adjustment per scenario
    eta = cp.Variable()                      # CVaR auxiliary (VaR)
    z = cp.Variable(S, nonneg=True)          # CVaR shortfall slacks

    # Vectorised: combined power Q[s,t] = q[t] + a[s,t]
    Q = cp.reshape(q, (1, H), order="C") + a   # broadcasts to (S, H)

    constraints = [Q <= cap, Q >= -cap]

    # SoC[s,t] = soc_init + cumsum_t (COP*Q[s,t]*0.25 - demand*0.25)
    drain = spec.baseline_demand_mw * 0.25
    cum = cp.cumsum(spec.cop * Q * 0.25 - drain, axis=1)
    soc_path = soc_init + cum
    constraints += [soc_path >= 0.0, soc_path <= spec.storage_mwh]

    # Per-scenario realised profit (DKK): sum over t of Q[s,t] * (da[t]-imb[s,t]) * 0.25
    spread = (da_path[None, :] - scen_imb) * 0.25  # (S, H)
    pl_per_scen = cp.sum(cp.multiply(Q, spread), axis=1)  # (S,)

    # CVaR on losses (loss = -profit)
    constraints += [z >= -pl_per_scen - eta]

    expected_profit = cp.sum(pl_per_scen) / S
    cvar_loss = eta + cp.sum(z) / (S * cfg.cvar_alpha)
    objective = cp.Maximize(expected_profit - cfg.cvar_lambda * cvar_loss)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)
    if q.value is None:
        return np.zeros(H)
    return np.clip(q.value, -cap, cap)


# ---------------------------------------------------------------------------
# Rolling-horizon evaluation harness (parallels evaluate_policy)
# ---------------------------------------------------------------------------


def evaluate_b3(panel: pl.DataFrame, spec: P2HSpec, cfg: B3Config) -> EvalResult:
    da = panel["da_price_dkk_mwh"].to_numpy()
    imb = panel["imbalance_price_dkk_mwh"].to_numpy()
    n = len(da)
    rng = np.random.default_rng(cfg.seed)

    soc = spec.initial_soc_mwh
    settlements = np.zeros(n, dtype=np.float64)
    qs = np.zeros(n, dtype=np.float64)
    bid_attempts = 0
    bid_violations = 0
    heat_unmet_ticks = 0
    energy_delivered = 0.0
    H = cfg.horizon_ticks
    residual_window_ticks = cfg.residual_window_days * TICKS_PER_DAY

    t0 = time.perf_counter()
    block_start = 0
    while block_start < n:
        block_end = min(block_start + H, n)
        Hblk = block_end - block_start
        # DA forecast for the horizon = realised DA (DA prices are known
        # day-ahead, so no forecasting noise on this leg — same as H&H).
        da_path = da[block_start:block_end]
        scen = _imb_scenarios(
            history_da=da[:block_start],
            history_imb=imb[:block_start],
            forecast_da=da_path,
            n_scenarios=cfg.n_scenarios,
            rng=rng,
            residual_window_ticks=residual_window_ticks,
        )
        q_block = _solve_horizon_lp(da_path, scen, soc, spec, cfg)
        for k in range(Hblk):
            t = block_start + k
            q_clamped = float(np.clip(q_block[k], -spec.capacity_mw, spec.capacity_mw))
            attempted = abs(q_clamped) > 1e-9
            if attempted:
                bid_attempts += 1
            soc_new = soc + spec.cop * q_clamped * 0.25 - spec.baseline_demand_mw * 0.25
            if soc_new < 0:
                if attempted and q_clamped < 0:
                    bid_violations += 1
                    q_clamped = max(
                        -spec.capacity_mw,
                        (spec.baseline_demand_mw * 0.25 - soc) / (spec.cop * 0.25),
                    )
                    q_clamped = min(q_clamped, 0.0)
                    soc_new = soc + spec.cop * q_clamped * 0.25 - spec.baseline_demand_mw * 0.25
                if soc_new < 0:
                    heat_unmet_ticks += 1
                    soc_new = 0.0
            if soc_new > spec.storage_mwh:
                if attempted and q_clamped > 0:
                    bid_violations += 1
                    q_clamped = max(
                        0.0,
                        (spec.storage_mwh - soc + spec.baseline_demand_mw * 0.25)
                        / (spec.cop * 0.25),
                    )
                    q_clamped = min(q_clamped, spec.capacity_mw)
                soc_new = min(
                    spec.storage_mwh,
                    soc + spec.cop * q_clamped * 0.25 - spec.baseline_demand_mw * 0.25,
                )
            settlement = q_clamped * (da[t] - imb[t]) * 0.25
            settlements[t] = settlement
            qs[t] = q_clamped
            energy_delivered += abs(q_clamped) * 0.25
            soc = soc_new
        block_start = block_end

    runtime = time.perf_counter() - t0
    total = float(settlements.sum())
    participation = float(np.mean(np.abs(qs) > 1e-9))
    sharpe = (
        float(settlements.mean() / settlements.std() * math.sqrt(ANNUAL_TICKS))
        if settlements.std() > 1e-9
        else 0.0
    )
    n_full_days = n // TICKS_PER_DAY
    daily = (
        settlements[: n_full_days * TICKS_PER_DAY]
        .reshape(n_full_days, TICKS_PER_DAY)
        .sum(axis=1)
    )
    cvar5 = float(np.sort(daily)[: max(1, int(np.ceil(0.05 * daily.size)))].mean()) if daily.size else 0.0
    profit_per_mwh = total / energy_delivered if energy_delivered > 1e-9 else 0.0
    return EvalResult(
        name="b3_hagstrom_herre",
        seed=cfg.seed,
        n_ticks=n,
        total_profit_dkk=total,
        profit_per_mwh_dkk=profit_per_mwh,
        sharpe=sharpe,
        cvar5_daily_dkk=cvar5,
        participation_rate=participation,
        physical_violation_rate=(bid_violations / bid_attempts) if bid_attempts else 0.0,
        runtime_seconds=runtime,
        extras={
            "energy_delivered_mwh": round(energy_delivered, 4),
            "ticks_with_bid": int(np.count_nonzero(np.abs(qs) > 1e-9)),
            "heat_unmet_tick_rate": round(heat_unmet_ticks / n, 4) if n else 0.0,
            "n_scenarios_per_lp": cfg.n_scenarios,
            "horizon_ticks": cfg.horizon_ticks,
            "cvar_alpha": cfg.cvar_alpha,
            "cvar_lambda": cfg.cvar_lambda,
        },
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


FROZEN_SEEDS = (13, 42, 137, 1729, 31415)


def run_b3(seeds=FROZEN_SEEDS, panel_split: str = "val") -> dict:
    cfg_template = B3Config()
    cfg_template.out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(panel_split)
    spec = P2HSpec()

    per_seed: dict[int, dict] = {}
    for s in seeds:
        cfg = B3Config(seed=s)
        result = evaluate_b3(panel, spec, cfg)
        per_seed[s] = result.to_json()
        (cfg_template.out_dir / f"b3_seed{s}.json").write_text(
            json.dumps(result.to_json(), indent=2)
        )
    keys = ["total_profit_eur", "profit_per_mwh_eur", "sharpe", "cvar5_daily_eur",
            "participation_rate", "physical_violation_rate"]
    rows = list(per_seed.values())
    agg = {
        k: {
            "mean": float(np.mean([r[k] for r in rows])),
            "std": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "values": [r[k] for r in rows],
        }
        for k in keys
    }
    out = {"per_seed": per_seed, "aggregate": agg}
    (cfg_template.out_dir / "aggregate.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=list(FROZEN_SEEDS))
    p.add_argument("--panel", choices=["val", "test"], default="val")
    args = p.parse_args()
    res = run_b3(seeds=tuple(args.seeds), panel_split=args.panel)
    print(json.dumps(res["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
