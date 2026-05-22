"""BoolQ evaluation: accuracy via 'Yes' vs 'No' next-token likelihood.

We don't sample/generate — we directly compare logprobs at the answer position.
This is the standard zero-shot / few-shot scoring approach for yes/no QA tasks
and avoids decoding nondeterminism.
"""
from __future__ import annotations

import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm

from dora_mini import data


@torch.no_grad()
def evaluate_boolq(model, tokenizer, dataset, max_length: int = 512) -> dict:
    """Compute BoolQ accuracy + average per-example NLL."""
    yes_ids = tokenizer("Yes", add_special_tokens=False)["input_ids"]
    no_ids = tokenizer("No", add_special_tokens=False)["input_ids"]
    yes_tok = yes_ids[0]
    no_tok = no_ids[0]

    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0
    nll_sum = 0.0

    for ex in tqdm(dataset, desc="eval"):
        formatted = data.format_for_eval(ex, tokenizer, max_length=max_length)
        input_ids = torch.tensor([formatted["input_ids"]], device=device)
        attention_mask = torch.tensor([formatted["attention_mask"]], device=device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        last_logits = out.logits[0, -1, :]
        logp = log_softmax(last_logits.float(), dim=-1)

        score_yes = logp[yes_tok].item()
        score_no = logp[no_tok].item()
        pred = 1 if score_yes > score_no else 0

        correct += int(pred == formatted["label"])
        total += 1
        # Absolute NLL here can look high (e.g. 10+) even when accuracy is ~88%: the model puts mass on other continuations (formatting tokens, spaces) before the actual yes/no token. The relative comparison above is what determines correctness — use accuracy as the headline metric.
        nll_sum += -(score_yes if formatted["label"] == 1 else score_no)

    return {"accuracy": correct / total, "loss": nll_sum / total}


@torch.no_grad()
def evaluate_boolq_generate(
    model,
    tokenizer,
    dataset,
    parser,
    gold_map,
    max_length: int = 512,
    max_new_tokens: int = 8,
) -> dict:
    """BoolQ accuracy via greedy generation + exact-match (the paper-style metric).

    `parser`: callable(text) -> extracted label string or None.
    `gold_map`: maps the BoolQ bool answer to the label string `parser` yields,
    e.g. {True: "yes", False: "no"} for a Yes/No-trained model.

    Unparseable generations score as wrong — this metric is sensitive to the
    format/training-stability collapse that likelihood scoring is blind to.
    """
    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0
    for ex in tqdm(dataset, desc="gen-eval"):
        formatted = data.format_for_eval(ex, tokenizer, max_length=max_length)
        input_ids = torch.tensor([formatted["input_ids"]], device=device)
        attention_mask = torch.tensor([formatted["attention_mask"]], device=device)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_text = tokenizer.decode(
            out[0, input_ids.shape[1]:], skip_special_tokens=True
        )
        pred = parser(gen_text)
        gold = gold_map[bool(ex["answer"])]
        correct += int(pred == gold)
        total += 1

    return {"genmatch_accuracy": correct / total}
