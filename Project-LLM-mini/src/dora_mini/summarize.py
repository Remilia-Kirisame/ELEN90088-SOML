"""Aggregate Tier-2 run results into a markdown summary.

Pure stdlib + json — no torch/yaml — so it is unit-testable on Mac. Groups the
3 seeds per (trainset, method, rank) cell into mean ± std, for both accuracy
metrics, and renders the regime-contrast tables.
"""
from __future__ import annotations

import json
import re
import statistics
from glob import glob

_RUN_ID = re.compile(
    r"^(?P<method>lora|dora)_mistral7b_(?P<trainset>boolq|cs170k)_r(?P<r>\d+)_s(?P<seed>\d+)$"
)


def parse_run_id(run_id: str) -> dict | None:
    """Parse a Tier-2 run id; None if it does not match (e.g. a Tier-1 dir)."""
    m = _RUN_ID.match(run_id)
    if not m:
        return None
    return {
        "method": m["method"],
        "trainset": m["trainset"],
        "r": int(m["r"]),
        "seed": int(m["seed"]),
    }


def load_runs(pattern: str = "results/**/metrics.json") -> list[dict]:
    """Load metrics.json for every completed Tier-2 seeded run.

    Tier-1 dirs (no _s suffix), unreadable/partial JSON, and runs missing an
    accuracy metric (e.g. the generation-eval pass has not run yet) are all
    skipped — summarize tolerates a partially-complete results/ directory.
    """
    runs = []
    for f in sorted(glob(pattern, recursive=True)):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not parse_run_id(d.get("run_id", "")):
            continue
        if "eval_accuracy_likelihood" in d and "eval_accuracy_genmatch" in d:
            runs.append(d)
    return runs


def _mean_std(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def aggregate(runs: list[dict]) -> dict[tuple, dict]:
    """Group runs by (trainset, method, rank) → aggregated cell stats."""
    groups: dict[tuple, list[dict]] = {}
    for d in runs:
        meta = parse_run_id(d["run_id"])
        groups.setdefault((meta["trainset"], meta["method"], meta["r"]), []).append(d)

    cells: dict[tuple, dict] = {}
    for key, ds in groups.items():
        lik_m, lik_s = _mean_std([d["eval_accuracy_likelihood"] for d in ds])
        gen_m, gen_s = _mean_std([d["eval_accuracy_genmatch"] for d in ds])
        rt_m, _ = _mean_std([d["train_runtime_s"] for d in ds])
        mem_m, _ = _mean_std([d["peak_memory_gb"] for d in ds])
        cells[key] = {
            "n": len(ds),
            "likelihood_mean": lik_m, "likelihood_std": lik_s,
            "genmatch_mean": gen_m, "genmatch_std": gen_s,
            "runtime_s_mean": rt_m, "peak_gb_mean": mem_m,
            "trainable_params": ds[0]["trainable_params"],
        }
    return cells


def _gap_table(cells: dict[tuple, dict], trainset: str, metric: str) -> list[str]:
    """DoRA - LoRA gap by rank for one regime and one metric (mean / std keys)."""
    out = [
        f"### {trainset} - DoRA - LoRA gap ({metric})",
        "",
        "| r | LoRA | DoRA | delta (DoRA - LoRA) |",
        "|---:|---:|---:|---:|",
    ]
    ranks = sorted({r for (ts, _, r) in cells if ts == trainset})
    for r in ranks:
        lo = cells.get((trainset, "lora", r))
        do = cells.get((trainset, "dora", r))
        if lo and do:
            lm, ls = lo[f"{metric}_mean"], lo[f"{metric}_std"]
            dm, ds = do[f"{metric}_mean"], do[f"{metric}_std"]
            out.append(
                f"| {r} | {lm:.4f} +/- {ls:.4f} | {dm:.4f} +/- {ds:.4f} | {dm - lm:+.4f} |"
            )
    out.append("")
    return out


def render(cells: dict[tuple, dict], zero_shot: dict | None) -> str:
    """Render the full markdown summary."""
    lines = ["# Tier 2 results summary", ""]
    if zero_shot:
        lines += [
            "## Zero-shot baseline (Mistral-7B-Instruct, no adapter)",
            "",
            f"- likelihood accuracy: {zero_shot.get('eval_accuracy_likelihood', float('nan')):.4f}",
            f"- genmatch accuracy:   {zero_shot.get('eval_accuracy_genmatch', float('nan')):.4f}",
            "",
        ]
    lines += ["## Per-cell results (mean +/- std over 3 seeds)", "",
              "| trainset | method | r | n | likelihood acc | genmatch acc | runtime (s) | peak mem (GB) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(cells):
        ts, method, r = key
        c = cells[key]
        lines.append(
            f"| {ts} | {method.upper()} | {r} | {c['n']} | "
            f"{c['likelihood_mean']:.4f} +/- {c['likelihood_std']:.4f} | "
            f"{c['genmatch_mean']:.4f} +/- {c['genmatch_std']:.4f} | "
            f"{c['runtime_s_mean']:.0f} | {c['peak_gb_mean']:.1f} |"
        )
    lines.append("")
    for trainset in ["boolq", "cs170k"]:
        if any(ts == trainset for (ts, _, _) in cells):
            lines += _gap_table(cells, trainset, "likelihood")
            lines += _gap_table(cells, trainset, "genmatch")
    lines += [
        "## How to regenerate", "",
        "```bash",
        "cd Project-LLM-mini && python scripts/summarize_results.py > results/SUMMARY.md",
        "```", "",
    ]
    return "\n".join(lines)


def build_summary() -> str:
    """Top-level: load everything under results/ and render the summary."""
    runs = load_runs()
    zero_shot = None
    zs_files = glob("results/**/zeroshot_*/metrics.json", recursive=True)
    if zs_files:
        with open(sorted(zs_files)[0]) as fh:
            zero_shot = json.load(fh)
    return render(aggregate(runs), zero_shot)
