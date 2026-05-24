#!/usr/bin/env python3
"""Aggregate run results into a markdown summary.

Run from the mini-project-DoRA root:

    python scripts/summarize_results.py --tier 2 > results/SUMMARY-tier2.md
    python scripts/summarize_results.py --tier 3 > results/SUMMARY-tier3.md

Default tier is 2 (the canonical project deliverable). Tier 3 is the 10k-step
cs170k enrichment; its renderer also loads Tier-2 cells for cross-tier comparison.
"""
import argparse
import sys
from pathlib import Path

# Make dora_mini importable when run directly (the package is not pip-installed on Mac).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dora_mini import summarize, summarize_tier3  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tier",
        type=int,
        default=2,
        choices=[2, 3],
        help="Which sweep to summarize: 2 (default, the canonical Tier-2 deliverable) or 3 (the 10k-step cs170k enrichment).",
    )
    args = p.parse_args()

    if args.tier == 2:
        if not summarize.load_runs():
            sys.stderr.write("no Tier-2 metrics.json found under results/\n")
            return 1
        print(summarize.build_summary())
    else:
        if not summarize_tier3.load_runs():
            sys.stderr.write("no Tier-3 metrics.json found under results/tier3-cs170k/\n")
            return 1
        print(summarize_tier3.build_summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
