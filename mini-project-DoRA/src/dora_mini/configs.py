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


# Optional trailing `_t3` marks a Tier-3 run; absence keeps Tier-2 routing untouched.
_GROUP_RE = re.compile(r"_(boolq|cs170k)_r\d+_s\d+(_t3)?$")


def run_group(run_id: str) -> str:
    """Map a run_id to its subfolder under configs/ and results/.

    Tier-2 seeded runs route by training regime (tier2-boolq / tier2-cs170k);
    Tier-3 runs (a `_t3` suffix appended after the seed token) route to tier3-<regime>;
    anything else — e.g. the unseeded Tier-1 run_ids — routes to tier1/.
    """
    m = _GROUP_RE.search(run_id)
    if not m:
        return "tier1"
    tier = "tier3" if m.group(2) else "tier2"
    return f"{tier}-{m.group(1)}"


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


# ---------------------------------------------------------------------------
# Tier-3 expansion: longer training (10k vs 2500 steps) + one extra seed,
# cs170k regime only (BoolQ is already saturated at 500 steps). Same grid
# shape as Tier-2's cs170k phase; designed to fit comfortably inside the
# gpu-h100 per-job time budget (DoRA 10k ≈ 6 h training + ≲ 30 min eval).
# Functions are parallel to the Tier-2 set so existing call sites and tests
# continue to see only the original 36-run grid.
# ---------------------------------------------------------------------------

TIER3_PHASES = {
    "cs170k": {
        "train_dataset": "commonsense_170k",
        "learning_rate": 2.0e-4,
        "num_steps": 10000,
        # Warmup scales proportionally with num_steps (4% of total, matching
        # Tier 2's 100/2500) so the LR-schedule shape is unchanged — Tier 3 is
        # Tier 2 stretched 4× in step count, not a different recipe.
        "warmup_steps": 400,
        # 25 in-training eval points across the run — same density per training
        # fraction as Tier 2 (100/2500 → 400/10000), keeping monitor-eval overhead
        # at the same ~7 min / run regardless of total length.
        "eval_every": 400,
    },
}
TIER3_SEEDS = [42, 1, 2, 3]


def build_tier3_config(method: str, rank: int, seed: int, phase: str = "cs170k") -> tuple[str, dict]:
    """Return (run_id, config-dict) for one Tier-3 run. Run ids carry a `_t3` suffix
    so `run_group()` routes outputs to `tier3-<regime>/` and Tier-2 results stay untouched."""
    recipe = TIER3_PHASES[phase]
    run_id = f"{method}_{MODEL_SHORT}_{phase}_r{rank}_s{seed}_t3"
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
            "train_size": None,
            "eval_size": 3270,
            "max_length": 512,
        },
        "training": {
            "batch_size": 4,
            "grad_accum": 4,
            "learning_rate": recipe["learning_rate"],
            "num_steps": recipe["num_steps"],
            "warmup_steps": recipe["warmup_steps"],
            "eval_every": recipe["eval_every"],
            "seed": seed,
        },
        "output": {"run_id": run_id},
    }
    return run_id, cfg


def all_tier3_configs() -> dict[str, dict]:
    """The 24-run Tier-3 sweep: 2 methods × 3 ranks × 4 seeds × 1 regime (cs170k)."""
    out: dict[str, dict] = {}
    for phase in TIER3_PHASES:
        for method in METHODS:
            for rank in RANKS:
                for seed in TIER3_SEEDS:
                    run_id, cfg = build_tier3_config(method, rank, seed, phase)
                    out[run_id] = cfg
    return out


def write_tier3_configs(configs_dir: Path) -> list[Path]:
    """Serialise every Tier-3 config to <configs_dir>/<group>/<run_id>.yaml. Returns paths written."""
    import yaml

    configs_dir = Path(configs_dir)
    written: list[Path] = []
    for run_id, cfg in sorted(all_tier3_configs().items()):
        path = configs_dir / run_group(run_id) / f"{run_id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        written.append(path)
    return written
