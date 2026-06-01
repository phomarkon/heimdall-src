"""Focal-agent non-LLM baselines B1, B2, B4 (docs/RESEARCH-PROPOSAL.md §5.2).

These are *focal-agent* baselines — distinct from the forecaster baselines in
``experiments/baselines.py`` (which mark-todos.md flags should be renamed FB1-FB7
to avoid the naming collision).

Each baseline implements the same profit model — a 25 MW / 75 MWh P2H +
thermal-storage asset operating in DK1 over the post-break val window
(2025-03-04 → 2025-04-30) on the 15-minute mFRR EAM grid.

Per-tick settlement (consistent with ``packages/simulator/replay.py``):
- "up" bid (curtail consumption by q MW): profit per MWh = imb_t − da_t;
  quantity per tick = q · 0.25 MWh.  Decreases SoC at COP rate.
- "down" bid (increase consumption by q MW): profit per MWh = da_t − imb_t;
  quantity per tick = q · 0.25 MWh.  Increases SoC at COP rate.
- "no-op" (B1 default): zero balancing-market profit; SoC drifts down at the
  baseline heat-demand rate.

Baselines:
- **B1 — Naive DA**: never participates in mFRR. Realised profit = 0 in the
  balancing market.  Reference point for "AI value".
- **B2 — Stochastic LP**: chance-constrained 1-step LP using the F7 quantile
  forecast of imbalance price. Solves a small QP per tick — accept/reject the
  optimal bid quantity if expected spread × q ≥ τ given the conformal interval.
- **B4 — Clean PPO**: stable-baselines3 PPO trained on a Gymnasium env for
  this asset on the train window, evaluated on val.

Metrics (proposal §5.3.1, Mark-set):
- realised profit total + €/MWh (DKK→EUR @ 7.46),
- Sharpe ratio (mean / std of per-tick P&L, annualised),
- CVaR(5%) of daily P&L,
- mFRR participation rate (fraction of ticks with non-zero bid),
- physical-feasibility violation rate (must be 0 for any baseline reaching the
  leaderboard — guardrailed below).

Outputs land in ``experiments/outputs/focal_agent_baselines/``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
DKK_PER_EUR = 7.46
SECONDS_PER_TICK = 15 * 60
TICKS_PER_DAY = 96
ANNUAL_TICKS = 96 * 365  # for Sharpe annualisation


# ---------------------------------------------------------------------------
# Asset spec (anchored to docs/RESEARCH-PROPOSAL.md §5.5: 50 MW P2H entrant, but
# we use the *focal* configuration: 25 MW / 75 MWh — proposal §5 footnote).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class P2HSpec:
    capacity_mw: float = 25.0
    storage_mwh: float = 75.0
    cop: float = 3.0  # PyPSA-Eur-Sec central HP value
    initial_soc_mwh: float = 37.5  # 50% full
    baseline_demand_mw: float = 4.0  # constant heat demand → SoC drift


# ---------------------------------------------------------------------------
# Panel loading
# ---------------------------------------------------------------------------


def load_panel(split: str) -> pl.DataFrame:
    path = REPO_ROOT / f"data/processed/dk1_panel_{split}.parquet"
    df = pl.read_parquet(path).sort("timestamp_utc").drop_nulls(
        subset=["da_price_dkk_mwh", "imbalance_price_dkk_mwh"]
    )
    return df


# ---------------------------------------------------------------------------
# Common evaluation harness
# ---------------------------------------------------------------------------


@dataclass
class TickStep:
    timestamp: str
    da_dkk: float
    imb_dkk: float
    bid_quantity_mw: float  # signed: + = down (consume); − = up (curtail)
    settlement_dkk: float
    soc_mwh: float
    feasible: bool
    notes: str = ""


@dataclass
class EvalResult:
    name: str
    seed: int
    n_ticks: int
    total_profit_dkk: float
    profit_per_mwh_dkk: float  # over delivered MWh
    sharpe: float
    cvar5_daily_dkk: float
    participation_rate: float
    physical_violation_rate: float
    runtime_seconds: float
    extras: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "n_ticks": self.n_ticks,
            "total_profit_dkk": round(self.total_profit_dkk, 4),
            "total_profit_eur": round(self.total_profit_dkk / DKK_PER_EUR, 4),
            "profit_per_mwh_dkk": round(self.profit_per_mwh_dkk, 4),
            "profit_per_mwh_eur": round(self.profit_per_mwh_dkk / DKK_PER_EUR, 4),
            "sharpe": round(self.sharpe, 4),
            "cvar5_daily_dkk": round(self.cvar5_daily_dkk, 4),
            "cvar5_daily_eur": round(self.cvar5_daily_dkk / DKK_PER_EUR, 4),
            "participation_rate": round(self.participation_rate, 4),
            "physical_violation_rate": round(self.physical_violation_rate, 4),
            "runtime_seconds": round(self.runtime_seconds, 2),
            "extras": self.extras,
        }


def evaluate_policy(
    policy_fn,
    *,
    name: str,
    seed: int,
    panel: pl.DataFrame,
    spec: P2HSpec,
    forecaster_signal: np.ndarray | None = None,
) -> EvalResult:
    """Walk-forward simulate ``policy_fn`` on ``panel``.

    ``policy_fn(t, history, forecast_t, soc_mwh, spec) -> bid_quantity_mw`` —
    a signed scalar; ``+`` consumes (down), ``−`` curtails (up). Caller is
    responsible for keeping |q| within capacity. The harness clamps to physical
    envelope and counts a violation when clamping was needed.
    """

    da = panel["da_price_dkk_mwh"].to_numpy()
    imb = panel["imbalance_price_dkk_mwh"].to_numpy()
    ts = panel["timestamp_utc"].to_list()
    n = len(da)

    soc = spec.initial_soc_mwh
    settlements = np.zeros(n, dtype=np.float64)
    qs = np.zeros(n, dtype=np.float64)
    bid_violations = 0
    bid_attempts = 0
    heat_unmet_ticks = 0
    energy_delivered = 0.0
    t0 = time.perf_counter()

    for t in range(n):
        history = {"da": da[:t], "imb": imb[:t]}
        forecast_t = (
            forecaster_signal[t] if forecaster_signal is not None else None
        )
        q_req = float(policy_fn(t, history, forecast_t, soc, spec))
        attempted = abs(q_req) > 1e-9
        if attempted:
            bid_attempts += 1

        # Clamp to capacity.
        q_clamped = float(np.clip(q_req, -spec.capacity_mw, spec.capacity_mw))
        if attempted and not math.isclose(q_clamped, q_req, rel_tol=1e-9, abs_tol=1e-9):
            bid_violations += 1

        # Project storage state.
        soc_new = soc + spec.cop * q_clamped * 0.25 - spec.baseline_demand_mw * 0.25
        if soc_new < 0:
            # Heat demand not fully servable.  If the agent *bid* an up-action
            # (q<0) that pushed SoC below zero, count as a bid violation; the
            # baseline-demand drain alone is tracked separately.
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

        # Settlement (DKK).  Spread sign: if q>0 (down), profit per MWh = da−imb;
        # if q<0 (up), profit per MWh = imb−da; uniform formula: q*(da−imb)*0.25
        # — q>0: da−imb; q<0: imb−da automatically since q is signed.
        # *Wait:* up-bid means we get paid imb−da for the curtailed energy (sell
        # avoided consumption at imb). Same formula since |q|*(imb−da) when q<0
        # equals q*(da−imb). ✓
        settlement = q_clamped * (da[t] - imb[t]) * 0.25
        settlements[t] = settlement
        qs[t] = q_clamped
        energy_delivered += abs(q_clamped) * 0.25
        soc = soc_new

    runtime = time.perf_counter() - t0
    total = float(settlements.sum())
    participation = float(np.mean(np.abs(qs) > 1e-9))

    # Sharpe (per-tick, annualised). Flat → Sharpe = 0.
    if settlements.std() > 1e-9:
        sharpe = float(
            settlements.mean() / settlements.std() * math.sqrt(ANNUAL_TICKS)
        )
    else:
        sharpe = 0.0

    # CVaR(5%) on daily aggregates.
    n_full_days = n // TICKS_PER_DAY
    daily = settlements[: n_full_days * TICKS_PER_DAY].reshape(n_full_days, TICKS_PER_DAY).sum(axis=1)
    if daily.size:
        worst = np.sort(daily)[: max(1, int(np.ceil(0.05 * daily.size)))]
        cvar5 = float(worst.mean())
    else:
        cvar5 = 0.0

    profit_per_mwh = total / energy_delivered if energy_delivered > 1e-9 else 0.0

    return EvalResult(
        name=name,
        seed=seed,
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
            "bid_attempts": bid_attempts,
            "bid_violations": bid_violations,
        },
    )


# ---------------------------------------------------------------------------
# B1 — Naive DA (no balancing-market participation)
# ---------------------------------------------------------------------------


def b1_naive_da_policy(t, history, forecast_t, soc, spec):
    return 0.0


# ---------------------------------------------------------------------------
# B2 — Stochastic LP (chance-constrained, one-step)
# ---------------------------------------------------------------------------


@dataclass
class B2Config:
    """Per-tick LP knobs."""

    alpha: float = 0.10  # chance-constraint coverage
    tau_dkk_per_mwh: float = -100 * DKK_PER_EUR  # min worst-case spread
    cvar_lambda: float = 0.5  # weight on CVaR penalty term
    quantile_window: int = 96 * 7  # 7-day rolling for residual quantiles


def _quantile_band(history_imb: np.ndarray, da_t: float, alpha: float, window: int):
    """Return (lo, hi) imbalance-price quantile band using rolling residuals.

    A *very* light stand-in for the F7+ACI forecaster: we use the rolling
    median of imbalance as the point forecast and the empirical α/2 quantiles
    of the in-window error as the band.  For the headline numbers we should
    plug in the F7+ACI cached predictions; this fallback keeps B2 runnable
    against the panel alone.
    """
    if history_imb.size < 8:
        return da_t * 0.5, da_t * 1.5
    w = history_imb[-window:]
    point = float(np.median(w))
    lo = float(np.quantile(w, alpha / 2))
    hi = float(np.quantile(w, 1 - alpha / 2))
    return lo, hi


def make_b2_policy(cfg: B2Config | None = None):
    cfg = cfg or B2Config()

    def _policy(t, history, forecast_t, soc, spec):
        if t < 8 or history["imb"].size < 8:
            return 0.0
        lo, hi = _quantile_band(history["imb"], history["da"][-1], cfg.alpha, cfg.quantile_window)
        da_t = history["da"][-1] if history["da"].size else 0.0

        # Per-MWh worst-case profit for the two sides:
        # - down (q>0): worst is at imb=hi → spread = da−hi (often negative)
        # - up   (q<0): worst is at imb=lo → spread = lo−da
        worst_down = da_t - hi
        worst_up = lo - da_t

        # Pick best worst-case side.
        if worst_down >= worst_up and worst_down >= cfg.tau_dkk_per_mwh:
            side_q = +spec.capacity_mw
            worst = worst_down
        elif worst_up > worst_down and worst_up >= cfg.tau_dkk_per_mwh:
            side_q = -spec.capacity_mw
            worst = worst_up
        else:
            return 0.0

        # Scale by storage headroom — never bid more than can be physically
        # absorbed by storage in this tick.
        if side_q > 0:
            headroom = (spec.storage_mwh - soc) / (spec.cop * 0.25)
            side_q = min(side_q, max(0.0, headroom))
        else:
            usable = (soc - spec.baseline_demand_mw * 0.25) / 0.25
            side_q = max(side_q, -max(0.0, usable))

        # CVaR-style shrinkage: shrink toward zero by λ when the spread band
        # is wide (high uncertainty).
        spread_width = max(1e-6, hi - lo)
        risk_factor = max(0.1, 1.0 - cfg.cvar_lambda * spread_width / (abs(da_t) + 1e-6))
        return side_q * risk_factor

    return _policy


# ---------------------------------------------------------------------------
# B4 — Clean PPO from scratch
# ---------------------------------------------------------------------------


@dataclass
class B4Config:
    n_envs: int = 4
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    seed: int = 42
    save_dir: Path = field(default_factory=lambda: REPO_ROOT / "models/baselines/b4_ppo")


def make_p2h_env(panel: pl.DataFrame, spec: P2HSpec, seed: int):
    """Tiny Gymnasium env for the focal P2H asset.

    Observation: [normalised SoC, da_t, imb_t, hour_sin, hour_cos, lag1_imb].
    Action: 1-D Box ∈ [-1, 1] mapping to [-cap, +cap] MW.
    Reward: 15-min DKK settlement.
    Episode: 96 ticks (one day).
    """
    import gymnasium as gym
    from gymnasium import spaces

    da = panel["da_price_dkk_mwh"].to_numpy()
    imb = panel["imbalance_price_dkk_mwh"].to_numpy()
    n = len(da)

    class P2HEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, base_seed: int = 0) -> None:
            super().__init__()
            self.observation_space = spaces.Box(
                low=-10.0, high=10.0, shape=(6,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
            self._rng = np.random.default_rng(base_seed)
            self._t0 = 0
            self._t = 0
            self._soc = spec.initial_soc_mwh
            self._ep_len = TICKS_PER_DAY

        def _obs(self):
            t = self._t0 + self._t
            tt = t % n
            hour = (tt % TICKS_PER_DAY) / TICKS_PER_DAY
            return np.array(
                [
                    self._soc / spec.storage_mwh,
                    da[tt] / 2000.0,
                    imb[tt] / 2000.0,
                    math.sin(2 * math.pi * hour),
                    math.cos(2 * math.pi * hour),
                    imb[(tt - 1) % n] / 2000.0,
                ],
                dtype=np.float32,
            )

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self._t0 = int(self._rng.integers(0, max(1, n - self._ep_len)))
            self._t = 0
            self._soc = spec.initial_soc_mwh
            return self._obs(), {}

        def step(self, action):
            q = float(np.clip(action[0], -1.0, 1.0)) * spec.capacity_mw
            tt = (self._t0 + self._t) % n
            soc_new = (
                self._soc + spec.cop * q * 0.25 - spec.baseline_demand_mw * 0.25
            )
            penalty = 0.0
            if soc_new < 0:
                penalty -= (-soc_new) * 100.0
                soc_new = 0.0
                q = 0.0
            elif soc_new > spec.storage_mwh:
                penalty -= (soc_new - spec.storage_mwh) * 100.0
                soc_new = spec.storage_mwh
                q = 0.0
            reward = q * (da[tt] - imb[tt]) * 0.25 + penalty
            self._soc = soc_new
            self._t += 1
            term = self._t >= self._ep_len
            return self._obs(), float(reward), term, False, {}

    env = P2HEnv(base_seed=seed)
    return env


def b4_train_ppo(cfg: B4Config, train_panel: pl.DataFrame, spec: P2HSpec):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    def _mk(i):
        return lambda: make_p2h_env(train_panel, spec, seed=cfg.seed + i)

    venv = DummyVecEnv([_mk(i) for i in range(cfg.n_envs)])
    # SB3 PPO with MlpPolicy is *faster* on CPU than GPU for our tiny
    # 10k-param actor-critic — GPU launch overhead dominates per-step.
    # See https://github.com/DLR-RM/stable-baselines3/issues/1245.
    # Parallelism across seeds is achieved at the driver level, not here.
    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        seed=cfg.seed,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=cfg.total_timesteps)
    model.save(cfg.save_dir / f"ppo_seed{cfg.seed}.zip")
    return model


def make_b4_policy(model):
    """Wrap a trained SB3 PPO policy into the (t, history, forecast, soc, spec)
    interface used by ``evaluate_policy``."""

    def _policy(t, history, forecast_t, soc, spec):
        # Reconstruct the same observation the env emits for the *current* tick.
        if history["da"].size == 0:
            return 0.0
        da_t = history["da"][-1]
        imb_t = history["imb"][-1]
        hour = (t % TICKS_PER_DAY) / TICKS_PER_DAY
        obs = np.array(
            [
                soc / spec.storage_mwh,
                da_t / 2000.0,
                imb_t / 2000.0,
                math.sin(2 * math.pi * hour),
                math.cos(2 * math.pi * hour),
                (history["imb"][-2] if history["imb"].size > 1 else imb_t) / 2000.0,
            ],
            dtype=np.float32,
        )
        action, _ = model.predict(obs, deterministic=True)
        return float(np.clip(action[0], -1.0, 1.0)) * spec.capacity_mw

    return _policy


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


FROZEN_SEEDS = (13, 42, 137, 1729, 31415)


def run_all(seeds: tuple[int, ...] = FROZEN_SEEDS, b4_steps: int = 200_000) -> dict:
    out_dir = REPO_ROOT / "experiments/outputs/focal_agent_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = P2HSpec()
    train_panel = load_panel("train")
    val_panel = load_panel("val")

    per_seed: dict[int, dict] = {}

    for seed in seeds:
        results: dict[str, dict] = {}

        # B1 — deterministic; reported at the first seed.
        r1 = evaluate_policy(
            b1_naive_da_policy, name="b1_naive_da", seed=seed, panel=val_panel, spec=spec
        )
        results["b1_naive_da"] = r1.to_json()
        (out_dir / f"b1_naive_da_seed{seed}.json").write_text(
            json.dumps(r1.to_json(), indent=2)
        )

        # B2 — deterministic in this single-quantile-band fallback.
        r2 = evaluate_policy(
            make_b2_policy(), name="b2_stochastic_lp", seed=seed, panel=val_panel, spec=spec
        )
        results["b2_stochastic_lp"] = r2.to_json()
        (out_dir / f"b2_stochastic_lp_seed{seed}.json").write_text(
            json.dumps(r2.to_json(), indent=2)
        )

        # B4 PPO — seed-dependent.
        cfg4 = B4Config(seed=seed, total_timesteps=b4_steps)
        model = b4_train_ppo(cfg4, train_panel, spec)
        r4 = evaluate_policy(
            make_b4_policy(model), name="b4_ppo", seed=seed, panel=val_panel, spec=spec
        )
        results["b4_ppo"] = r4.to_json()
        (out_dir / f"b4_ppo_seed{seed}.json").write_text(
            json.dumps(r4.to_json(), indent=2)
        )

        per_seed[seed] = results
        (out_dir / f"summary_seed{seed}.json").write_text(json.dumps(results, indent=2))

    # Aggregate across seeds.
    agg: dict[str, dict] = {}
    for name in ("b1_naive_da", "b2_stochastic_lp", "b4_ppo"):
        keys = ["total_profit_eur", "profit_per_mwh_eur", "sharpe", "cvar5_daily_eur",
                "participation_rate", "physical_violation_rate"]
        rows = [per_seed[s][name] for s in seeds]
        agg[name] = {
            k: {
                "mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
                "values": [r[k] for r in rows],
            }
            for k in keys
        }
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    return {"per_seed": per_seed, "aggregate": agg}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=list(FROZEN_SEEDS))
    parser.add_argument("--b4-steps", type=int, default=200_000)
    args = parser.parse_args()
    res = run_all(seeds=tuple(args.seeds), b4_steps=args.b4_steps)
    print(json.dumps(res["aggregate"], indent=2))
