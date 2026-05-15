"""End-to-end training: read YAML, train, save adapter, write metrics.json.

The structured stdout follows spec §11 (env / config / model / per-step /
final blocks) so train.log is self-readable when committed to git.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import torch
import yaml
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from dora_mini import data, models, paths


def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False, "name": "cpu"}
    p = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": p.name,
        "memory_gb": p.total_memory / 1e9,
        "capability": f"{p.major}.{p.minor}",
    }


def _print_env() -> None:
    import peft
    import transformers

    g = _gpu_info()
    print("=== env ===", flush=True)
    print(f"python  : {platform.python_version()}")
    print(f"torch   : {torch.__version__}")
    print(f"transformers : {transformers.__version__}")
    print(f"peft    : {peft.__version__}")
    if g["available"]:
        print(
            f"device  : cuda:0 ({g['name']}, cc {g['capability']}, "
            f"{g['memory_gb']:.1f} GB)"
        )
    else:
        print("device  : cpu")
    print(
        f"slurm   : job {os.environ.get('SLURM_JOB_ID', 'interactive')} "
        f"on {platform.node()}"
    )
    print("==========", flush=True)


def _print_config(config: dict) -> None:
    print("=== config ===", flush=True)
    print(yaml.safe_dump(config, sort_keys=False).rstrip())
    print("==========", flush=True)


def _print_model(cfg: dict, model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        mem_alloc = mem_total = 0.0
    print("=== model ===", flush=True)
    print(f"base       : {cfg['model']['name']}")
    print(
        f"peft       : {cfg['peft']['method'].upper()} "
        f"(r={cfg['peft']['r']}, alpha={cfg['peft']['alpha']}, "
        f"target={cfg['peft']['target_modules']})"
    )
    print(f"trainable  : {trainable:,} / {total:,}  ({100*trainable/total:.3f}%)")
    print(f"mem before : {mem_alloc:.1f} / {mem_total:.1f} GB (gpu allocated)")
    print("==========", flush=True)


class LossHistoryCallback(TrainerCallback):
    """Capture per-step training loss for metrics.json."""

    def __init__(self) -> None:
        self.curve: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            self.curve.append(
                {"step": state.global_step, "loss": float(logs["loss"])}
            )


def run_training(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    run_id = cfg["output"]["run_id"]
    rdir = paths.results_dir(run_id)
    (rdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    _print_env()
    _print_config(cfg)

    tokenizer = models.load_tokenizer(cfg["model"]["name"])
    base = models.load_base_model(cfg["model"]["name"], cfg["model"]["dtype"])
    model = models.wrap_peft(base, **cfg["peft"])
    _print_model(cfg, model)

    train_ds = data.load_boolq("train", limit=cfg["data"].get("train_size"))
    eval_ds = data.load_boolq("validation", limit=cfg["data"].get("eval_size"))
    max_len = cfg["data"]["max_length"]

    def _fmt_train(ex):
        return data.format_for_training(ex, tokenizer, max_length=max_len)

    train_ds = train_ds.map(_fmt_train, remove_columns=train_ds.column_names)
    eval_for_loss = eval_ds.map(_fmt_train, remove_columns=eval_ds.column_names)

    args = TrainingArguments(
        output_dir=str(rdir / "trainer_out"),
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["grad_accum"],
        learning_rate=cfg["training"]["learning_rate"],
        max_steps=cfg["training"]["num_steps"],
        warmup_steps=cfg["training"]["warmup_steps"],
        logging_steps=cfg["training"]["eval_every"],
        eval_steps=cfg["training"]["eval_every"],
        eval_strategy="steps",
        save_strategy="no",
        bf16=(cfg["model"]["dtype"] == "bfloat16"),
        seed=cfg["training"]["seed"],
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=False,
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    history = LossHistoryCallback()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_for_loss,
        data_collator=collator,
        callbacks=[history],
    )

    started = time.time()
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    print(f"=== training start {started_iso} ===", flush=True)
    trainer.train()
    finished = time.time()
    finished_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished))
    peak_mem_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    )
    print(
        f"=== training end {finished_iso} ({finished-started:.0f} s) ===",
        flush=True,
    )

    model.save_pretrained(rdir / "adapter")

    # BoolQ accuracy eval (yes/no likelihood)
    from dora_mini.eval import evaluate_boolq

    eval_metrics = evaluate_boolq(model, tokenizer, eval_ds, max_length=max_len)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    import peft
    import transformers

    metrics = {
        "run_id": run_id,
        "config_resolved": cfg,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "gpu": _gpu_info().get("name", "cpu"),
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "interactive"),
        "started_at": started_iso,
        "finished_at": finished_iso,
        "trainable_params": trainable,
        "trainable_pct": 100 * trainable / total,
        "peak_memory_gb": peak_mem_gb,
        "train_runtime_s": int(finished - started),
        "loss_curve": history.curve,
        "eval_accuracy": eval_metrics["accuracy"],
        "eval_loss": eval_metrics["loss"],
    }
    (rdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("=== final ===", flush=True)
    print(f"eval_accuracy : {eval_metrics['accuracy']:.4f}")
    print(f"eval_loss     : {eval_metrics['loss']:.4f}")
    print(f"train_runtime : {int(finished - started)} s")
    print(f"peak_mem      : {peak_mem_gb:.1f} GB")
    print(f"trainable_params : {trainable}")
    print(f"slurm_job_id  : {os.environ.get('SLURM_JOB_ID', 'interactive')}")
    print("==========", flush=True)
