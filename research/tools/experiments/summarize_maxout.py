"""Max-out forecaster leaderboard.

Scans every ``models/forecaster/<MODEL>/seed-*/`` directory (excluding the
``canonical_*`` and ``*_lo_*`` runs, which belong to the apples-to-apples
sub-table) and aggregates val pinball + coverage per model, mean ± std over
seeds. Falls back to recomputing pinball from ``val_preds.npz`` when a
trainer didn't write ``metrics.json``.

The output is a "best forecaster" leaderboard for each target inferred from
the directory name (``*_activation*`` → activation, else → price).

Usage:
    uv run python tools/experiments/summarize_maxout.py \\
        [--root models/forecaster] [--out outputs/maxout_summary]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Skip these subtrees — they belong to other tables.
EXCLUDE_PREFIXES = ("canonical_", "f5_canonical_", "f6_canonical_")
EXCLUDE_FILES = {"README.md", "summary_all.json", "sweep_summary.json", "coverage_audit.json"}
EXCLUDE_DIRS = {"f3_lite"}  # appendix-only DeepARLite (proposal note ADR-0006)

ACTIVATION_PAT = re.compile(r"(?:^activation_|_activation(?:_|$))")


def _pinball_from_metrics(m: dict) -> float | None:
    for k in ("val_pinball_mean_dkk", "val_pinball_mean"):
        v = m.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _pinball_from_npz(npz: Path, quantiles=(0.1, 0.5, 0.9)) -> tuple[float, float] | None:
    try:
        import numpy as np  # noqa: PLC0415
        d = np.load(npz)
        preds, targets = d["preds"], d["targets"]
        if preds.ndim != 3 or targets.ndim != 2 or preds.shape[2] != len(quantiles):
            return None
        per = []
        for qi, q in enumerate(quantiles):
            err = targets - preds[..., qi]
            per.append(float(np.mean(np.maximum(q * err, (q - 1.0) * err))))
        srt = np.sort(preds, axis=-1)
        cov = float(np.mean((targets >= srt[..., 0]) & (targets <= srt[..., -1])))
        return float(sum(per) / len(per)), cov
    except Exception:
        return None


def _scan_model(model_dir: Path) -> dict:
    pinballs: list[float] = []
    coverages: list[float] = []
    acis: list[float] = []
    widths: list[float] = []
    seeds_present: list[int] = []
    for sd in sorted(model_dir.glob("seed-*/")):
        seed_str = sd.name.removeprefix("seed-")
        try:
            seeds_present.append(int(seed_str))
        except ValueError:
            continue
        mfile = sd / "metrics.json"
        if mfile.exists():
            data = json.loads(mfile.read_text())
            pin = _pinball_from_metrics(data)
            if pin is not None:
                pinballs.append(pin)
            for k, lst in (
                ("val_q10_q90_coverage", coverages),
                ("aci_empirical_coverage", acis),
                ("aci_mean_width", widths),
            ):
                v = data.get(k)
                if isinstance(v, (int, float)):
                    lst.append(float(v))
            continue
        npz = sd / "val_preds.npz"
        if npz.exists():
            fb = _pinball_from_npz(npz)
            if fb is not None:
                pin, cov = fb
                pinballs.append(pin)
                coverages.append(cov)

    def _agg(vs: list[float]) -> tuple[float, float]:
        if not vs:
            return float("nan"), float("nan")
        if len(vs) == 1:
            return float(vs[0]), 0.0
        return float(stats.mean(vs)), float(stats.pstdev(vs))

    pin_mu, pin_sd = _agg(pinballs)
    cov_mu, _ = _agg(coverages)
    aci_mu, _ = _agg(acis)
    wid_mu, _ = _agg(widths)
    return {
        "n_seeds": len(seeds_present),
        "pinball_mean": pin_mu, "pinball_std": pin_sd,
        "raw_cov_mean": cov_mu,
        "aci_cov_mean": aci_mu, "aci_width_mean": wid_mu,
    }


def collect(root: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name in EXCLUDE_DIRS:
            continue
        if any(sub.name.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if "_lo_" in sub.name:
            continue
        # Must contain at least one seed-* subdir.
        if not any(sub.glob("seed-*/")):
            continue
        rows[sub.name] = _scan_model(sub)
    return rows


def _classify_target(name: str) -> str:
    if ACTIVATION_PAT.search(name):
        return "activation"
    return "price"


def _table(rows: dict[str, dict], target: str) -> str:
    head = ("| Model | n_seeds | Val pinball | Raw cov | ACI cov | ACI width |\n"
            "|---|---:|---:|---:|---:|---:|\n")
    lines = []
    items = [(k, v) for k, v in rows.items() if _classify_target(k) == target]
    items.sort(key=lambda kv: (float("inf") if (kv[1].get("pinball_mean") != kv[1].get("pinball_mean")) else kv[1]["pinball_mean"]))
    for name, r in items:
        if r["n_seeds"] == 0:
            continue
        pm, ps = r["pinball_mean"], r["pinball_std"]
        cv, ac, wd = r["raw_cov_mean"], r["aci_cov_mean"], r["aci_width_mean"]
        lines.append(
            f"| {name} | {r['n_seeds']} | {pm:.1f} ± {ps:.1f} | "
            f"{cv:.2f} | {ac:.2f} | {wd:.1f} |"
        )
    return f"### Target: {target}\n\n" + head + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=REPO_ROOT / "models/forecaster")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "outputs/maxout_summary")
    args = p.parse_args(argv)
    rows = collect(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    md = "# Max-out forecaster leaderboard\n\n"
    md += "Each model trained with its own tuned config + best feature set " \
          "(NOT the apples-to-apples F_CANONICAL panel). One row per model.\n\n"
    for t in ("price", "activation"):
        md += _table(rows, t) + "\n"
    args.out.with_suffix(".md").write_text(md)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out.with_suffix('.md')}, .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
