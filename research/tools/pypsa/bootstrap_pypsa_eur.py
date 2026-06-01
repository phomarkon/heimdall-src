from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.pypsa_adapter import PyPSAEurConfig  # noqa: E402


PYPSA_EUR_URL = "https://github.com/PyPSA/pypsa-eur.git"


def resolve_target_dir(target_dir: Path | None) -> Path:
    if target_dir is not None:
        return target_dir.expanduser().resolve()
    return PyPSAEurConfig.from_env().pypsa_eur_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a pinned PyPSA-Eur checkout")
    parser.add_argument("--ref", default="v2026.02.0")
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(
            f"PyPSA-Eur bootstrap dry-run: clone {PYPSA_EUR_URL} "
            f"into {resolve_target_dir(args.target_dir)} at {args.ref}"
        )
        return

    target_dir = resolve_target_dir(args.target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if not target_dir.exists():
        subprocess.run(
            ["git", "clone", PYPSA_EUR_URL, str(target_dir)],
            check=True,
        )
    subprocess.run(["git", "-C", str(target_dir), "fetch", "--tags"], check=True)
    subprocess.run(["git", "-C", str(target_dir), "checkout", args.ref], check=True)
    print(f"PyPSA-Eur ready at {target_dir} ({args.ref})")


if __name__ == "__main__":
    main()
