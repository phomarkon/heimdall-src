"""Build the thesis-ready canonical comparison table from saved metrics.

Scans ``models/forecaster/canonical_<model>_<target>/seed-<S>/metrics.json``,
aggregates mean ± std across seeds, and emits both Markdown (for quick
inspection) and LaTeX (for chapter 03). One row per (model, target).

Usage:
    uv run python tools/experiments/summarize_canonical.py \\
        [--root models/forecaster] [--out outputs/canonical_summary]

Output:
    outputs/canonical_summary.md
    outputs/canonical_summary.tex
    outputs/canonical_summary.json  (machine-readable for further plotting)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

CANONICAL_DIR_RE = re.compile(r"^canonical_(?P<model>.+)_(?P<target>price|activation)$")
METRIC_KEYS = (
    "val_pinball_mean", "val_pinball_mean_dkk",      # one or the other present
    "val_q10_q90_coverage", "aci_empirical_coverage", "aci_mean_width",
)


def _pinball(m: dict) -> float | None:
    return m.get("val_pinball_mean_dkk") or m.get("val_pinball_mean")


def _agg(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(stats.mean(values)), float(stats.pstdev(values))


def _pinball_from_npz(npz_path: Path, quantiles=(0.1, 0.5, 0.9)) -> tuple[float, float] | None:
    """Recompute (mean pinball, q10-q90 coverage) from a val_preds.npz file.
    Used as a fallback for trainers that don't write metrics.json (patchTST).
    Returns None if the file is missing or malformed.
    """
    try:
        import numpy as np  # noqa: PLC0415
        d = np.load(npz_path)
        preds, targets = d["preds"], d["targets"]
        per_q = []
        for qi, q in enumerate(quantiles):
            err = targets - preds[..., qi]
            per_q.append(float(np.mean(np.maximum(q * err, (q - 1.0) * err))))
        srt = np.sort(preds, axis=-1)
        cov = float(np.mean((targets >= srt[..., 0]) & (targets <= srt[..., -1])))
        return float(sum(per_q) / len(per_q)), cov
    except Exception:
        return None


def collect(root: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    for sub in sorted(root.glob("canonical_*_*/")):
        m = CANONICAL_DIR_RE.match(sub.name)
        if not m:
            continue
        model, target = m["model"], m["target"]
        seed_dirs = sorted(sub.glob("seed-*/"))
        per_seed: dict[str, list[float]] = defaultdict(list)
        for sd in seed_dirs:
            mfile = sd / "metrics.json"
            if mfile.exists():
                data = json.loads(mfile.read_text())
                pinball = _pinball(data)
                if pinball is not None:
                    per_seed["pinball"].append(pinball)
                for k in ("val_q10_q90_coverage", "aci_empirical_coverage", "aci_mean_width"):
                    v = data.get(k)
                    if isinstance(v, (int, float)):
                        per_seed[k].append(float(v))
                continue
            npz = sd / "val_preds.npz"
            if npz.exists():
                fb = _pinball_from_npz(npz)
                if fb is not None:
                    pin, cov = fb
                    per_seed["pinball"].append(pin)
                    per_seed["val_q10_q90_coverage"].append(cov)
                continue
        row = {"n_seeds": len(seed_dirs)}
        for k, vs in per_seed.items():
            mu, sd = _agg(vs)
            row[f"{k}_mean"], row[f"{k}_std"] = mu, sd
        rows[(model, target)] = row
    return rows


def _md_table(rows: dict[tuple[str, str], dict], target: str) -> str:
    head = ("| Model | n_seeds | Val pinball (mean ± std) | Raw q10–q90 cov "
            "| ACI cov | ACI width |\n|---|---:|---:|---:|---:|---:|\n")
    lines = []
    for (model, t), row in sorted(rows.items()):
        if t != target:
            continue
        pin = row.get("pinball_mean", float("nan"))
        ps = row.get("pinball_std", 0.0)
        raw = row.get("val_q10_q90_coverage_mean", float("nan"))
        aci = row.get("aci_empirical_coverage_mean", float("nan"))
        wid = row.get("aci_mean_width_mean", float("nan"))
        lines.append(
            f"| {model} | {row['n_seeds']} | {pin:.1f} ± {ps:.1f} | "
            f"{raw:.2f} | {aci:.2f} | {wid:.1f} |"
        )
    return f"### Target: {target}\n\n" + head + "\n".join(lines) + "\n"


def _latex_table(rows: dict[tuple[str, str], dict], target: str) -> str:
    body = []
    for (model, t), row in sorted(rows.items()):
        if t != target:
            continue
        pin = row.get("pinball_mean", float("nan"))
        ps = row.get("pinball_std", 0.0)
        raw = row.get("val_q10_q90_coverage_mean", float("nan"))
        aci = row.get("aci_empirical_coverage_mean", float("nan"))
        wid = row.get("aci_mean_width_mean", float("nan"))
        body.append(
            f"{model.replace('_', r'\_')} & {row['n_seeds']} & "
            f"${pin:.1f}\\pm{ps:.1f}$ & {raw:.2f} & {aci:.2f} & {wid:.1f} \\\\"
        )
    return (
        "\\begin{table}[t]\n"
        f"\\caption{{Canonical apples-to-apples comparison on \\texttt{{F\\_CANONICAL}} "
        f"(44 features), target = {target}. Mean $\\pm$ std over five seeds "
        "$\\{13, 42, 137, 1729, 31415\\}$.}\n"
        f"\\label{{tab:canonical-{target}}}\n"
        "\\centering\n"
        "\\begin{tabular}{@{}llrrrr@{}}\n"
        "\\toprule\n"
        "Model & $n$ & Val pinball & Raw cov & ACI cov & ACI width \\\\\n"
        "\\midrule\n"
        + "\n".join(body) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=REPO_ROOT / "models/forecaster")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/canonical_summary")
    args = p.parse_args(argv)

    rows = collect(args.root)
    if not rows:
        print(f"no canonical_*/seed-* metrics found under {args.root}")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)

    md = "# Canonical apples-to-apples forecaster comparison\n\n"
    for target in ("price", "activation"):
        md += _md_table(rows, target) + "\n"
    args.out.with_suffix(".md").write_text(md)

    latex = "\n\n".join(_latex_table(rows, t) for t in ("price", "activation"))
    args.out.with_suffix(".tex").write_text(latex)

    serialisable = {f"{m}__{t}": v for (m, t), v in rows.items()}
    args.out.with_suffix(".json").write_text(json.dumps(serialisable, indent=2))
    print(f"wrote {args.out.with_suffix('.md')}, .tex, .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
