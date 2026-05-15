"""CLI: python scripts/train.py --config configs/foo.yaml"""
from __future__ import annotations

import argparse
import sys

from dora_mini.train import run_training


def main() -> int:
    p = argparse.ArgumentParser(description="Train a PEFT adapter on BoolQ.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    args = p.parse_args()
    run_training(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
