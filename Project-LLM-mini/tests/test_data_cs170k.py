"""Tests for the commonsense_170k loader and formatter."""
import json

import pytest

from dora_mini import data


class StubTokenizer:
    """Deterministic char-level tokenizer — one token id per character."""

    eos_token = "#"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<" + messages[0]["content"] + ">"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(c) for c in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_format_commonsense_masks_prompt():
    tok = StubTokenizer()
    ex = {"instruction": "Q?", "input": "", "output": "A"}
    out = data.format_commonsense_for_training(ex, tok, max_length=512)
    prompt_text = "<Q?>"                       # apply_chat_template
    n_prompt = len(prompt_text)                # char-level => one id per char
    # prompt tokens masked, answer tokens kept
    assert out["labels"][:n_prompt] == [-100] * n_prompt
    assert out["labels"][n_prompt:] == out["input_ids"][n_prompt:]
    # full sequence = prompt + answer + eos
    assert out["input_ids"] == [ord(c) for c in "<Q?>A#"]


def test_format_commonsense_appends_input():
    tok = StubTokenizer()
    ex = {"instruction": "Q?", "input": "ctx", "output": "A"}
    out = data.format_commonsense_for_training(ex, tok, max_length=512)
    assert out["input_ids"] == [ord(c) for c in "<Q?\n\nctx>A#"]


def test_load_commonsense170k_reads_json(tmp_path):
    pytest.importorskip("datasets")
    p = tmp_path / "cs.json"
    p.write_text(json.dumps([
        {"instruction": "i1", "input": "", "output": "o1", "answer": "true"},
        {"instruction": "i2", "input": "", "output": "o2", "answer": "false"},
    ]))
    ds = data.load_commonsense170k(p)
    assert len(ds) == 2
    assert ds[0]["instruction"] == "i1"
    assert len(data.load_commonsense170k(p, limit=1)) == 1
