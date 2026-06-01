"""Regenerate Heimdall thesis figures from source.

This is the reproducibility entry point for every diagram in
``manuscript/figures/02_se`` (Chapter 2, software-engineering).
Three classes of figure live in that directory:

* PlantUML hand-authored diagrams under ``puml/``. We render each ``.puml`` to
  ``<name>.pdf`` next to ``puml/``.
* Code-generated UML class diagrams via ``pyreverse`` (ships with pylint).
  We target the contract package because it pins the inter-service wire
  format and is the most useful single class diagram for the reader.
* A package dependency graph via ``pydeps``.

The script is intentionally a single file with no third-party Python deps
beyond what pylint / pydeps bring in via ``uvx``. Run it from the repo root::

    make figures-se          # default: 02_se only
    uv run scripts/build_figures.py --all     # also 01_market, 03_ml

Every external tool invocation is shown on stdout so a reviewer can see what
happened. Missing tools produce a clear error rather than a silent skip.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SE_DIR = REPO_ROOT / "manuscript" / "figures" / "02_se"
PUML_DIR = SE_DIR / "puml"
GENERATED_DIR = SE_DIR / "generated"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run ``cmd``, echoing it first, and exit on failure."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        sys.exit(f"command failed (exit {result.returncode}): {' '.join(cmd)}")


def require(tool: str, hint: str) -> str:
    """Return ``tool``'s absolute path or exit with a fix-it hint."""
    path = shutil.which(tool)
    if path is None:
        sys.exit(f"missing tool: {tool}\nfix: {hint}")
    return path


def render_plantuml() -> int:
    """Render every ``*.puml`` under ``puml/`` to a PDF next to ``puml/``.

    We go via SVG and convert to PDF with ``rsvg-convert`` because PlantUML's
    native ``-tpdf`` backend silently drops all text in this toolchain. SVG keeps
    the labels and stays vector for LaTeX.
    """
    plantuml = require(
        "plantuml",
        "brew install plantuml  # or download the jar from plantuml.com",
    )
    rsvg = require("rsvg-convert", "brew install librsvg")
    sources = sorted(p for p in PUML_DIR.glob("*.puml") if not p.name.startswith("_"))
    if not sources:
        print(f"no .puml sources in {PUML_DIR}")
        return 0
    run([plantuml, "-tsvg", "-o", str(SE_DIR.resolve()), *map(str, sources)])
    for src in sources:
        svg = SE_DIR / f"{src.stem}.svg"
        pdf = SE_DIR / f"{src.stem}.pdf"
        if svg.exists():
            run([rsvg, "-f", "pdf", "-o", str(pdf), str(svg)])
            svg.unlink()
    return len(sources)


def render_class_diagram() -> None:
    """Run pyreverse on the contracts package and convert .dot to PDF."""
    GENERATED_DIR.mkdir(exist_ok=True)
    dot = require("dot", "brew install graphviz")
    # pyreverse is shipped by pylint; we invoke it via ``uvx`` to keep the
    # workspace clean.
    uv = require("uv", "see https://docs.astral.sh/uv/")
    target = REPO_ROOT / "packages" / "contracts" / "src" / "heimdall_contracts"
    run(
        [
            uv,
            "tool",
            "run",
            "--from",
            "pylint",
            "pyreverse",
            "--output",
            "dot",
            "--output-directory",
            str(GENERATED_DIR),
            "--project",
            "heimdall_contracts",
            str(target),
        ]
    )
    # pyreverse writes classes_<project>.dot and packages_<project>.dot.
    for stem in ("classes_heimdall_contracts", "packages_heimdall_contracts"):
        dot_path = GENERATED_DIR / f"{stem}.dot"
        if dot_path.exists():
            pdf_path = SE_DIR / f"{stem.replace('heimdall_', '')}.pdf"
            run([dot, "-Tpdf", str(dot_path), "-o", str(pdf_path)])


def render_pydeps() -> None:
    """Render an inter-package dependency graph via pydeps."""
    uv = require("uv", "see https://docs.astral.sh/uv/")
    out = SE_DIR / "package_dependency_graph.svg"
    pkg_root = REPO_ROOT / "packages"
    # ``--no-show`` skips opening a browser; ``--cluster`` groups by package.
    run(
        [
            uv,
            "tool",
            "run",
            "--from",
            "pydeps",
            "pydeps",
            str(pkg_root),
            "--no-show",
            "--cluster",
            "--max-bacon",
            "4",
            "-o",
            str(out),
        ]
    )
    # Convert SVG → PDF for LaTeX inclusion (use cairosvg via uvx if rsvg-convert
    # is unavailable).
    rsvg = shutil.which("rsvg-convert")
    pdf = out.with_suffix(".pdf")
    if rsvg:
        run([rsvg, "-f", "pdf", "-o", str(pdf), str(out)])
    else:
        run([uv, "tool", "run", "--from", "cairosvg", "cairosvg", str(out), "-o", str(pdf)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-puml", action="store_true")
    parser.add_argument("--skip-class", action="store_true")
    parser.add_argument("--skip-pydeps", action="store_true")
    args = parser.parse_args()

    if not args.skip_puml:
        n = render_plantuml()
        print(f"rendered {n} PlantUML diagrams → {SE_DIR}")
    if not args.skip_class:
        render_class_diagram()
        print(f"rendered pyreverse class + package diagrams → {SE_DIR}")
    if not args.skip_pydeps:
        render_pydeps()
        print(f"rendered package dependency graph → {SE_DIR}")


if __name__ == "__main__":
    main()
