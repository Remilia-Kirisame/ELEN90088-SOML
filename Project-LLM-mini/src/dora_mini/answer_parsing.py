"""Answer extraction for the generation-based exact-match metric.

Pure stdlib (`re`) — deliberately torch-free so it is unit-testable on Mac and
importable by both eval.py and scripts/evaluate.py without the CUDA stack.
"""
from __future__ import annotations

import re

_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_TRUE_FALSE = re.compile(r"\b(true|false)\b", re.IGNORECASE)
_BOOLEAN = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)


def parse_yes_no(text: str) -> str | None:
    """First whole-word 'yes'/'no' in `text`, lowercased. None if absent."""
    m = _YES_NO.search(text)
    return m.group(1).lower() if m else None


def parse_true_false(text: str) -> str | None:
    """First whole-word 'true'/'false' in `text`, lowercased. None if absent."""
    m = _TRUE_FALSE.search(text)
    return m.group(1).lower() if m else None


def parse_yes_no_or_true_false(text: str) -> str | None:
    """First whole-word yes/no/true/false in `text`, normalized to 'true'/'false'.

    Used for cs170k-trained models: cs170k training expects true/false but the
    BoolQ eval prompt instructs yes/no — low-rank adapters that obey the prompt
    would otherwise score 0 even when answering correctly. Maps yes->true,
    no->false so the metric measures task accuracy regardless of which vocabulary
    the model emits.
    """
    m = _BOOLEAN.search(text)
    if not m:
        return None
    word = m.group(1).lower()
    return "true" if word in {"yes", "true"} else "false"
