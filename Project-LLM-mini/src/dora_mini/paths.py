"""Resolve project paths from environment variables.

Site-specific values (`PROJECT_DIR`, `UV_PROJECT_ENVIRONMENT`, `HF_HOME`) are
set by `scripts/local/setup_env.sh` (gitignored) or the OOD pre-run block on
Spartan. This module fails fast if anything is missing — better than silently
defaulting to the wrong path.
"""
from __future__ import annotations

import os
from pathlib import Path


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(
            f"{name} is not set. Source scripts/local/setup_env.sh first."
        )
    return val


def project_dir() -> Path:
    """Root of the mini-project working tree (e.g. ~/ELEN90088-SOML/Project-LLM-mini)."""
    return Path(_require_env("PROJECT_DIR"))


def venv_dir() -> Path:
    """uv-managed venv path; set via UV_PROJECT_ENVIRONMENT."""
    return Path(_require_env("UV_PROJECT_ENVIRONMENT"))


def hf_home() -> Path:
    """HuggingFace cache root."""
    return Path(_require_env("HF_HOME"))


def results_dir(run_id: str) -> Path:
    """Per-run results directory; mkdir parents=True exist_ok=True (idempotent — preserves existing contents)."""
    if not run_id or not run_id.strip():
        raise ValueError("run_id must not be empty")
    d = project_dir() / "results" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
