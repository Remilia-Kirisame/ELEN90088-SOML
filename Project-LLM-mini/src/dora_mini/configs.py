"""Tier-2 sweep definition — the single source of truth for all experiment configs.

`all_configs()` builds the 36-run sweep as plain dicts (pure, stdlib-only).
`write_configs()` serialises them to standalone YAML files. Each emitted file is
self-contained: `python scripts/train.py --config configs/<name>.yaml`.
"""
from __future__ import annotations

import re
from pathlib import Path

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "mistral7b"
METHODS = ["lora", "dora"]
RANKS = [4, 8, 16]
SEEDS = [42, 1, 2]

# Per-phase recipe. The key is the run-id <trainset> token.
PHASES = {
    "boolq": {
        "train_dataset": "boolq",
        "learning_rate": 5.0e-5,
        "num_steps": 500,
        "warmup_steps": 50,
    },
    "cs170k": {
        "train_dataset": "commonsense_170k",
        "learning_rate": 2.0e-4,
        # Finalised at 2500 by the Phase-2 timing probe (loss flattens by step ~1,200).
        "num_steps": 2500,
        "warmup_steps": 100,
    },
}


_GROUP_RE = re.compile(r"_(boolq|cs170k)_r\d+_s\d+$")


def run_group(run_id: str) -> str:
    """Map a run_id to its subfolder under configs/ and results/.

    Tier-2 seeded runs route by training regime (tier2-boolq / tier2-cs170k);
    anything else — e.g. the unseeded Tier-1 run_ids — routes to tier1/.
    """
    m = _GROUP_RE.search(run_id)
    return f"tier2-{m.group(1)}" if m else "tier1"


def build_config(method: str, rank: int, seed: int, phase: str) -> tuple[str, dict]:
    """Return (run_id, config-dict) for one run."""
    recipe = PHASES[phase]
    run_id = f"{method}_{MODEL_SHORT}_{phase}_r{rank}_s{seed}"
    cfg = {
        "model": {"name": MODEL, "dtype": "bfloat16"},
        "peft": {
            "method": method,
            "r": rank,
            "alpha": 2 * rank,
            "target_modules": "all-linear",
            "dropout": 0.05,
        },
        "data": {
            "train_dataset": recipe["train_dataset"],
            "train_size": None,            # null = full pool
            "eval_size": 3270,             # full BoolQ dev
            "max_length": 512,
        },
        "training": {
            "batch_size": 4,
            "grad_accum": 4,
            "learning_rate": recipe["learning_rate"],
            "num_steps": recipe["num_steps"],
            "warmup_steps": recipe["warmup_steps"],
            "eval_every": 100,
            "seed": seed,
        },
        "output": {"run_id": run_id},
    }
    return run_id, cfg


def all_configs() -> dict[str, dict]:
    """The full 36-run sweep, keyed by run_id."""
    out: dict[str, dict] = {}
    for phase in PHASES:
        for method in METHODS:
            for rank in RANKS:
                for seed in SEEDS:
                    run_id, cfg = build_config(method, rank, seed, phase)
                    out[run_id] = cfg
    return out


def write_configs(configs_dir: Path) -> list[Path]:
    """Serialise every config to <configs_dir>/<group>/<run_id>.yaml. Returns paths written."""
    import yaml  # function-level so dict-building tests need no yaml

    configs_dir = Path(configs_dir)
    written: list[Path] = []
    for run_id, cfg in sorted(all_configs().items()):
        path = configs_dir / run_group(run_id) / f"{run_id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        # yaml.safe_dump emits floats as 5.0e-05 (dotted) — YAML-1.1-safe.
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        written.append(path)
    return written
