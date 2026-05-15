"""BoolQ dataset loading and chat-template formatting.

We use the instruction-tuned model's chat template (via the tokenizer) so the
formatting works for any HF model with a chat template defined (Mistral-Instruct,
Phi-Instruct, etc.). For training we compute prompt-length and mask the prompt
tokens with -100 so loss is only on the answer token(s). For eval we keep the
prompt-only form and score 'Yes' vs 'No' likelihoods.
"""
from __future__ import annotations

from typing import Any, Mapping

from datasets import Dataset, load_dataset


_BOOLQ_INSTRUCTION = (
    "Based on the following passage, answer the question with a single word: "
    "yes or no.\n\nPassage: {passage}\nQuestion: {question}"
)


def load_boolq(split: str, limit: int | None = None) -> Dataset:
    """Load BoolQ split. Cached under HF_DATASETS_CACHE (set via env)."""
    ds = load_dataset("boolq", split=split)
    if limit is not None and limit < len(ds):
        ds = ds.select(range(limit))
    return ds


def format_for_training(
    example: Mapping[str, Any], tokenizer, max_length: int = 512
) -> dict[str, list[int]]:
    """Format one BoolQ example for causal-LM training.

    Returns input_ids + attention_mask + labels, where prompt tokens are masked
    (-100) so the loss only flows through the 'Yes'/'No' answer tokens.
    """
    user_msg = _BOOLQ_INSTRUCTION.format(
        passage=example["passage"], question=example["question"]
    )
    answer = "Yes" if example["answer"] else "No"

    # Tokenize the prompt-only form (with generation prompt) to find its length.
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Tokenize the full sequence (prompt + answer + eos).
    full_text = prompt_text + answer + tokenizer.eos_token
    full = tokenizer(
        full_text, truncation=True, max_length=max_length, add_special_tokens=False
    )

    labels = list(full["input_ids"])
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100

    return {
        "input_ids": full["input_ids"],
        "attention_mask": full["attention_mask"],
        "labels": labels,
    }


def format_for_eval(
    example: Mapping[str, Any], tokenizer, max_length: int = 512
) -> dict[str, Any]:
    """Format one BoolQ example as prompt-only for likelihood scoring.

    Returns input_ids + attention_mask + the integer label (1=yes, 0=no).
    """
    user_msg = _BOOLQ_INSTRUCTION.format(
        passage=example["passage"], question=example["question"]
    )
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
    out = tokenizer(
        prompt_text, truncation=True, max_length=max_length, add_special_tokens=False
    )
    return {
        "input_ids": out["input_ids"],
        "attention_mask": out["attention_mask"],
        "label": int(example["answer"]),
    }
