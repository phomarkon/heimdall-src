"""Enable `python -m heimdall_ai_society <command>` (used by SOCIETY-PLAN and the batch runners)."""

from __future__ import annotations

import sys

from heimdall_ai_society.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
