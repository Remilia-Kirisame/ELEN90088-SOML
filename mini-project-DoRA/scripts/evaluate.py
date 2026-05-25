"""Generation exact-match eval pass over saved adapters (or the base model).

Adapter run — append eval_accuracy_genmatch to an existing metrics.json:
    python scripts/evaluate.py --run results/<group>/<run_id>

Zero-shot baseline — score the base model, no adapter, write a fresh metrics.json:
    python scripts/evaluate.py --zero-shot --model mistralai/Mistral-7B-Instruct-v0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from peft import PeftModel

from dora_mini import data, models
from dora_mini.answer_parsing import parse_yes_no, parse_yes_no_or_true_false
from dora_mini.eval import evaluate_boolq, evaluate_boolq_generate

# Each phase's parser is chosen to accept the vocabulary the model is likely to
# emit. BoolQ models reliably say yes/no. cs170k models are trained to say
# true/false, but the BoolQ eval prompt instructs yes/no — low-rank cs170k
# adapters obey the prompt rather than overriding it with training, so we accept
# both vocabularies (normalized to true/false) to keep the metric a measure of
# task accuracy rather than format-adaptation strength.
_PARSERS = {
    "boolq": (parse_yes_no, {True: "yes", False: "no"}),
    "commonsense_170k": (parse_yes_no_or_true_false, {True: "true", False: "false"}),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", help="results/<group>/<run_id> dir with config.yaml + adapter/.")
    p.add_argument("--zero-shot", action="store_true", help="Score the base model, no adapter.")
    p.add_argument("--model", help="Base model name (zero-shot only).")
    p.add_argument("--eval-size", type=int, default=3270)
    p.add_argument("--debug-print", type=int, default=0, help="Print first N raw generations to stderr; skips the metrics.json write (debug mode, non-destructive).")
    args = p.parse_args()

    if not args.zero_shot and args.run is None:
        p.error("one of --run or --zero-shot is required")
    if args.zero_shot and args.model is None:
        p.error("--model is required with --zero-shot")

    if args.zero_shot:
        model_name = args.model
        tokenizer = models.load_tokenizer(model_name)
        model = models.load_base_model(model_name, "bfloat16")
        parser, gold_map = _PARSERS["boolq"]   # untuned instruct model answers yes/no
        run_dir = Path("results/tier2-baseline/zeroshot_mistral7b_boolq")
        # The zero-shot baseline lives at a fixed path under tier2-baseline/. It is
        # the Tier-2 deliverable's reference line — re-running here overwrites it
        # in place. Surface that to the user before the (expensive) eval pass starts.
        if (run_dir / "metrics.json").exists() and not args.debug_print:
            print(
                f"WARNING: zero-shot metrics.json already exists at {run_dir / 'metrics.json'}; "
                "this run will overwrite the Tier-2 baseline. Re-run with --debug-print N to inspect "
                "without writing, or move the existing file out of the way to preserve it.",
                file=sys.stderr,
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics: dict = {"run_id": run_dir.name, "model": model_name,
                         "eval_size": args.eval_size}
        eval_ds = data.load_boolq("validation", limit=args.eval_size)
        lik = evaluate_boolq(model, tokenizer, eval_ds, max_length=512)
        metrics["eval_accuracy_likelihood"] = lik["accuracy"]
        metrics["eval_loss"] = lik["loss"]
    else:
        run_dir = Path(args.run)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        model_name = cfg["model"]["name"]
        tokenizer = models.load_tokenizer(model_name)
        base = models.load_base_model(model_name, cfg["model"]["dtype"])
        model = PeftModel.from_pretrained(base, str(run_dir / "adapter"))
        parser, gold_map = _PARSERS[cfg["data"]["train_dataset"]]
        metrics = json.loads((run_dir / "metrics.json").read_text())
        eval_ds = data.load_boolq("validation", limit=args.eval_size)

    gen = evaluate_boolq_generate(model, tokenizer, eval_ds, parser, gold_map, max_length=512, debug_print=args.debug_print)
    metrics["eval_accuracy_genmatch"] = gen["genmatch_accuracy"]
    if args.debug_print > 0:
        print(f"{run_dir.name}: genmatch_accuracy (debug, n={args.eval_size}) = {gen['genmatch_accuracy']:.4f}  [metrics.json NOT written]", file=sys.stderr)
    else:
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"{run_dir.name}: genmatch_accuracy = {gen['genmatch_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
