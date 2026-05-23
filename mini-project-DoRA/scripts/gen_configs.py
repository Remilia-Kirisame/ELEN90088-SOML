#!/usr/bin/env python3
"""Generate experiment configs. Run from the mini-project-DoRA root:

    python scripts/gen_configs.py             # Tier 2 (default): 36 configs
    python scripts/gen_configs.py --tier 3    # Tier 3: 24 cs170k configs (10k steps, 4 seeds)
"""
import argparse
import sys
from pathlib import Path

# Make dora_mini importable when run directly (the package is not pip-installed on Mac).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dora_mini.configs import write_configs, write_tier3_configs  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", type=int, default=2, choices=[2, 3],
                   help="Which sweep to generate: 2 (default, 36 runs) or 3 (24 runs, cs170k-only).")
    args = p.parse_args()

    if args.tier == 2:
        written = write_configs(Path("configs"))
    else:
        written = write_tier3_configs(Path("configs"))
    print(f"wrote {len(written)} configs to configs/ (tier {args.tier})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
