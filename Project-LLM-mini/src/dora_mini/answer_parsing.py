"""Answer extraction for the generation-based exact-match metric.

Pure stdlib (`re`) — deliberately torch-free so it is unit-testable on Mac and
importable by both eval.py and scripts/evaluate.py without the CUDA stack.
"""
from __future__ import annotations

import re

_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_TRUE_FALSE = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def parse_yes_no(text: str) -> str | None:
    """First whole-word 'yes'/'no' in `text`, lowercased. None if absent."""
    m = _YES_NO.search(text)
    return m.group(1).lower() if m else None


def parse_true_false(text: str) -> str | None:
    """First whole-word 'true'/'false' in `text`, lowercased. None if absent."""
    m = _TRUE_FALSE.search(text)
    return m.group(1).lower() if m else None
