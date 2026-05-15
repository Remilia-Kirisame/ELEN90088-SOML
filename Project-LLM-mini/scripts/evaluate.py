"""CLI: re-evaluate a saved adapter against BoolQ.

python scripts/evaluate.py --adapter results/<run_id>/adapter/ --config results/<run_id>/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from peft import PeftModel

from dora_mini import data, models
from dora_mini.eval import evaluate_boolq


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="Path to PEFT adapter directory.")
    p.add_argument("--config", required=True, help="Path to that run's config.yaml.")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tokenizer = models.load_tokenizer(cfg["model"]["name"])
    base = models.load_base_model(cfg["model"]["name"], cfg["model"]["dtype"])
    model = PeftModel.from_pretrained(base, args.adapter)

    eval_ds = data.load_boolq("validation", limit=cfg["data"].get("eval_size"))
    metrics = evaluate_boolq(
        model, tokenizer, eval_ds, max_length=cfg["data"]["max_length"]
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
