"""Minimal env + model sanity check. Target runtime under 5 minutes on A100.

Exercises every component once:
  - tokenizer load
  - base model load (bfloat16)
  - PEFT wrap (use_dora=True, r=8)
  - format 4 BoolQ examples
  - one forward + backward pass
  - one optimizer step

Prints env + model summary blocks. No checkpoint saved.
"""
from __future__ import annotations

import sys

import torch
from transformers import DataCollatorForSeq2Seq

from dora_mini import data, models
from dora_mini.train import _print_env


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


def main() -> int:
    _print_env()

    print(f"\n--- step 1/6: load tokenizer ({MODEL_NAME})")
    tokenizer = models.load_tokenizer(MODEL_NAME)
    print(f"vocab size: {tokenizer.vocab_size}, eos: {tokenizer.eos_token!r}")

    print(f"\n--- step 2/6: load base model in bfloat16")
    base = models.load_base_model(MODEL_NAME, "bfloat16")
    print(f"loaded. mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(f"\n--- step 3/6: wrap with PEFT (DoRA r=8)")
    model = models.wrap_peft(
        base, method="dora", r=8, alpha=16,
        target_modules="all-linear", dropout=0.05,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    print(f"\n--- step 4/6: load 4 BoolQ examples and format")
    ds = data.load_boolq("train", limit=4)
    batch = [data.format_for_training(ex, tokenizer, max_length=256) for ex in ds]
    print(f"got {len(batch)} examples; input_ids lens: "
          f"{[len(b['input_ids']) for b in batch]}")

    print(f"\n--- step 5/6: one forward+backward pass")
    collator = DataCollatorForSeq2Seq(tokenizer)
    inputs = collator(batch)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model(**inputs)
    print(f"loss: {out.loss.item():.4f}")
    out.loss.backward()
    grad_norm = (
        sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None)
        ** 0.5
    )
    print(f"grad norm (PEFT params only): {grad_norm:.4f}")

    print(f"\n--- step 6/6: one optimizer step")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    opt.step()
    opt.zero_grad()
    print("optimizer step OK")

    print(f"\n=== smoke test passed ===")
    print(f"peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
