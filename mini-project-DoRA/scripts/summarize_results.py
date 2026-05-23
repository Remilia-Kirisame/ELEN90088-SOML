#!/usr/bin/env python3
"""Aggregate Tier-2 run results into results/SUMMARY.md.

Run from the mini-project-DoRA root:

    python scripts/summarize_results.py > results/SUMMARY.md
"""
import sys
from pathlib import Path

# Make dora_mini importable when run directly (the package is not pip-installed on Mac).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dora_mini.summarize import build_summary, load_runs  # noqa: E402


def main() -> int:
    if not load_runs():
        sys.stderr.write("no Tier-2 metrics.json found under results/\n")
        return 1
    print(build_summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
