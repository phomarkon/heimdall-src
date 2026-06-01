"""KE1/KE2/KE3 verdict-summary figure (LaTeX-friendly PNG).

Per docs/RESEARCH-PROPOSAL.md §5.6 the three kill experiments gate the project at
sprint days 2/9/14. A single figure shows the verdict + the headline number
for each.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from heimdall_ml.viz import apply_paper_style
from heimdall_ml.viz.style import PAGE_WIDTH_IN

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTES = REPO_ROOT / "notes"
FIG_DIR = REPO_ROOT / "figures"


def _ke1_summary() -> tuple[str, str]:
    txt = (NOTES / "ke1_verdict.md").read_text()
    verdict = "PASS" if "Verdict: PASS" in txt else "FAIL"
    m = re.search(r"is (\d+\.\d+)% below", txt)
    detail = f"F7 beats EWMA by {m.group(1)} % pinball" if m else "F7 vs EWMA"
    return verdict, detail


def _ke2_summary() -> tuple[str, str]:
    txt = (NOTES / "ke2_verdict.md").read_text()
    verdict = "PASS" if "Verdict: PASS" in txt else "FAIL"
    m = re.search(r"D = \*\*(\d+\.\d+)\*\*", txt)
    detail = f"KS D={m.group(1)}" if m else "KS test"
    return verdict, detail


def _ke3_summary() -> tuple[str, str]:
    txt = (NOTES / "ke3_verdict.md").read_text()
    verdict = "PASS" if "Verdict: PASS" in txt else "FAIL"
    m = re.search(r"B200 \+ ACI\s*\|\s*\d+\.\d+\s*\|\s*(\d+\.\d+)", txt)
    detail = f"GPU+ACI p99 = {m.group(1)} ms" if m else "latency p99"
    return verdict, detail


def main() -> int:
    apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, fn, sec in [
        ("KE1 (volatility)", _ke1_summary, "§5.6 day 2"),
        ("KE2 (data quality)", _ke2_summary, "§5.6 day 9"),
        ("KE3 (latency)", _ke3_summary, "§5.6 day 14"),
    ]:
        try:
            v, d = fn()
        except FileNotFoundError:
            v, d = "n/a", "verdict file missing"
        rows.append((name, sec, v, d))

    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 1.6))
    ax.axis("off")
    cell_text = [[r[0], r[1], r[2], r[3]] for r in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Kill experiment", "Sprint", "Verdict", "Headline"],
        loc="center",
        cellLoc="left",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    # Colour PASS rows green, FAIL red.
    for i, r in enumerate(rows, start=1):
        for j in range(4):
            cell = table[i, j]
            if r[2] == "PASS":
                cell.set_facecolor("#D6F5DC")
            elif r[2] == "FAIL":
                cell.set_facecolor("#F7D6D6")

    fig.suptitle("Kill-experiment verdicts (RESEARCH-PROPOSAL §5.6)", y=0.96)
    fig.savefig(FIG_DIR / "ke_verdict_table.png")
    plt.close(fig)
    print(f"-> {FIG_DIR / 'ke_verdict_table.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
