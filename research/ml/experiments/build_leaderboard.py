"""Aggregate every model's val_preds + metrics into a single leaderboard.

Reads ``models/forecaster/<name>/seed-*/metrics.json`` and emits
``notes/forecaster_leaderboard.md`` with rows F0/F3/F7/F8/F9/B1..B7 and columns
q10 / q50 / q90 pinball (mean ± std), ACI cov, p99 latency where available.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "models/forecaster"


def _read_seed_metrics(model_dir: Path) -> list[dict]:
    out = []
    for seed_dir in sorted(model_dir.glob("seed-*")):
        m = seed_dir / "metrics.json"
        if m.exists():
            d = json.loads(m.read_text())
            out.append(d)
    return out


def _agg(values: list[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{mean(values):.1f} ± {stdev(values):.1f}"


def main() -> int:
    rows: list[dict] = []
    for model_dir in sorted(MODEL_ROOT.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        if model_dir.name.startswith("seed-") or model_dir.name == "release":
            continue
        per_seed = _read_seed_metrics(model_dir)
        if not per_seed:
            continue
        rows.append(
            {
                "name": model_dir.name,
                "n_seeds": len(per_seed),
                "q10": _agg([d["val_pinball_q10"] for d in per_seed if "val_pinball_q10" in d]),
                "q50": _agg([d["val_pinball_q50"] for d in per_seed if "val_pinball_q50" in d]),
                "q90": _agg([d["val_pinball_q90"] for d in per_seed if "val_pinball_q90" in d]),
                "mean": _agg([d.get("val_pinball_mean", d.get("val_pinball_mean_dkk")) for d in per_seed]),
                "raw_cov": _agg([d.get("val_q10_q90_coverage", float("nan")) * 100 for d in per_seed]),
                "aci_cov": _agg(
                    [d["aci_empirical_coverage"] * 100 for d in per_seed if "aci_empirical_coverage" in d]
                ),
            }
        )
    # Sort by q50 mean (just take the mean of the mean column for sorting key).
    def _key(row):
        try:
            return float(row["q50"].split("±")[0].strip())
        except Exception:
            return 1e9

    rows.sort(key=_key)

    ke3_path = REPO_ROOT / "notes/ke3_verdict.md"
    p99_note = ""
    if ke3_path.exists():
        p99_note = "Latency (p99) numbers are F7-only; see `notes/ke3_verdict.md`."

    body = [
        "# Forecaster Zoo Leaderboard",
        "",
        "Per docs/RESEARCH-PROPOSAL.md §5.3.1 the published numbers average over the "
        "frozen seed set `[13, 42, 137, 1729, 31415]`. Pinball loss in DKK/MWh "
        "(lower = better). Coverage in % (target 90 %). Raw cov = empirical "
        "[q10,q90] coverage of the model's quantile heads; ACI cov = empirical "
        "coverage after wrapping the q50 with online ACI (Theorem 1b).",
        "",
        "| Model | n seeds | q10 pinball | q50 pinball | q90 pinball | mean pinball | raw cov (%) | ACI cov (%) |",
        "|-------|--------:|------------:|------------:|------------:|-------------:|------------:|------------:|",
    ]
    for r in rows:
        body.append(
            f"| {r['name']} | {r['n_seeds']} | {r['q10']} | {r['q50']} | {r['q90']} | {r['mean']} | {r['raw_cov']} | {r['aci_cov']} |"
        )
    body.append("")
    body.append(p99_note)

    out = REPO_ROOT / "notes/forecaster_leaderboard.md"
    out.write_text("\n".join(body))
    print(f"leaderboard -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
