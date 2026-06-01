"""Build the leave-one-group-out (LOGO) delta table.

Compares each ``canonical_<model>_<target>_lo_<group>/seed-*/metrics.json``
against the matching baseline ``canonical_<model>_<target>/seed-*/metrics.json``
and reports the mean Δpinball per group, averaged over seeds.

A positive Δ means dropping the group *hurts* (the group carried signal);
a near-zero or negative Δ means the group was redundant or noise.

Usage:
    uv run python tools/experiments/summarize_logo.py \\
        --model f8 --target price [--root models/forecaster]

Output:
    outputs/logo_<model>_<target>.md  /  .tex  /  .json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as stats
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LO_RE = re.compile(r"^canonical_(?P<model>.+?)_(?P<target>price|activation)_lo_(?P<group>.+)$")


def _pinball(m: dict) -> float | None:
    return m.get("val_pinball_mean_dkk") or m.get("val_pinball_mean")


def _pinball_from_npz(npz_path: Path, quantiles=(0.1, 0.5, 0.9)) -> float | None:
    """Fallback for trainers (patchTST) that don't write metrics.json."""
    try:
        import numpy as np  # noqa: PLC0415
        d = np.load(npz_path)
        preds, targets = d["preds"], d["targets"]
        per_q = []
        for qi, q in enumerate(quantiles):
            err = targets - preds[..., qi]
            per_q.append(float(np.mean(np.maximum(q * err, (q - 1.0) * err))))
        return float(sum(per_q) / len(per_q))
    except Exception:
        return None


def _mean_pinball(directory: Path) -> tuple[float, float, int]:
    vs: list[float] = []
    for sd_dir in sorted(directory.glob("seed-*/")):
        mfile = sd_dir / "metrics.json"
        if mfile.exists():
            v = _pinball(json.loads(mfile.read_text()))
            if isinstance(v, (int, float)):
                vs.append(float(v))
            continue
        npz = sd_dir / "val_preds.npz"
        if npz.exists():
            v = _pinball_from_npz(npz)
            if isinstance(v, (int, float)):
                vs.append(float(v))
    if not vs:
        return float("nan"), float("nan"), 0
    mu = stats.mean(vs)
    sd = stats.pstdev(vs) if len(vs) > 1 else 0.0
    return mu, sd, len(vs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--target", choices=["price", "activation"], required=True)
    p.add_argument("--root", type=Path, default=REPO_ROOT / "models/forecaster")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "outputs/logo_summary")
    args = p.parse_args(argv)

    baseline = args.root / f"canonical_{args.model}_{args.target}"
    mu_b, sd_b, n_b = _mean_pinball(baseline)
    if n_b == 0:
        print(f"baseline not found: {baseline}")
        return 1

    rows = []
    for sub in sorted(args.root.glob(f"canonical_{args.model}_{args.target}_lo_*/")):
        m = LO_RE.match(sub.name)
        if not m:
            continue
        group = m["group"]
        mu, sd, n = _mean_pinball(sub)
        if n == 0:
            continue
        delta = mu - mu_b
        rows.append({"group": group, "pinball_mean": mu, "pinball_std": sd,
                     "n_seeds": n, "delta_vs_baseline": delta})

    rows.sort(key=lambda r: -r["delta_vs_baseline"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.model}_{args.target}"

    md = (f"# LOGO ablation — {args.model} / {args.target}\n\n"
          f"Baseline (F_CANONICAL, n={n_b} seeds): "
          f"**{mu_b:.1f} ± {sd_b:.1f}**\n\n"
          "| Dropped group | Pinball (mean ± std) | Δ vs baseline | n |\n"
          "|---|---:|---:|---:|\n")
    md += "\n".join(
        f"| {r['group']} | {r['pinball_mean']:.1f} ± {r['pinball_std']:.1f} "
        f"| {r['delta_vs_baseline']:+.1f} | {r['n_seeds']} |"
        for r in rows
    ) + "\n"
    (args.out.with_suffix("").with_name(args.out.name + suffix + ".md")).write_text(md)

    (args.out.with_suffix("").with_name(args.out.name + suffix + ".json")).write_text(
        json.dumps({"baseline": {"mean": mu_b, "std": sd_b, "n": n_b}, "rows": rows}, indent=2)
    )
    print(f"wrote {args.out.name}{suffix}.md and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
