"""CLI wrapper around dora_mini.train.run_training.

Use: python scripts/train.py --config configs/<run>.yaml
Submit via SLURM: sbatch scripts/sbatch_train.sh configs/<run>.yaml
The heavy lifting (HF Trainer integration, metrics writing) is in dora_mini/train.py — this file is just argparse + delegation.
"""
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
