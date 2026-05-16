"""Model + tokenizer loading and PEFT wrapping.

The single switch between LoRA and DoRA is `LoraConfig(use_dora=True)` from
HuggingFace PEFT (available since 0.9). Same code path, same target modules,
same rank — only the use_dora flag flips.
"""
from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(model_name: str, dtype: str = "bfloat16"):
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype: {dtype}. Choose from {list(_DTYPE_MAP)}.")
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=_DTYPE_MAP[dtype],
        device_map="auto",
    )


def wrap_peft(
    model,
    *,
    method: str,
    r: int,
    alpha: int,
    target_modules,
    dropout: float,
):
    """Wrap a base model with LoRA (use_dora=False) or DoRA (use_dora=True)."""
    if method not in ("lora", "dora"):
        raise ValueError(f"Unknown peft method: {method}")
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=(method == "dora"),
    )
    return get_peft_model(model, config)
