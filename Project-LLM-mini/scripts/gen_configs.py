#!/usr/bin/env python3
"""Generate the 36 Tier-2 experiment configs. Run from the Project-LLM-mini root:

    python scripts/gen_configs.py
"""
import sys
from pathlib import Path

# Make dora_mini importable when run directly (the package is not pip-installed on Mac).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dora_mini.configs import write_configs  # noqa: E402


def main() -> int:
    written = write_configs(Path("configs"))
    print(f"wrote {len(written)} configs to configs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
