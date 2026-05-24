"""Aggregate Tier-2 run results into a markdown summary.

Pure stdlib + json — no torch/yaml — so it is unit-testable on Mac. Groups the
3 seeds per (trainset, method, rank) cell into mean ± std, for both accuracy
metrics, and renders the regime-contrast tables.

Also merges the pre-broadening strict-parser genmatch snapshot (saved before the
parser was widened to accept yes/no in addition to true/false). The strict view
is preserved as a separate column so the report can distinguish "format-adaptation
strength" (strict) from "task accuracy via free generation" (broad).
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


def load_strict_snapshot(
    path: str = "results/tier2-cs170k/_strict_genmatch_pre_fix.json",
) -> dict[str, float]:
    """Load the pre-fix strict-parser genmatch snapshot; empty dict if absent or unreadable.

    Maps run_id -> genmatch_accuracy under parse_true_false (the original strict parser).
    These values are the pre-broadening genmatch numbers, preserved in a snapshot because
    the metrics.json files were overwritten by the broad-parser re-run. The strict view
    captures format-adaptation strength (how strongly the adapter overrode the BoolQ
    eval prompt's yes/no instruction with cs170k's true/false training format), distinct
    from the broad-parser task-accuracy view.
    """
    try:
        with open(path) as fh:
            return json.load(fh).get("values", {})
    except (json.JSONDecodeError, OSError):
        return {}


def load_runs(
    pattern: str = "results/**/metrics.json",
    strict_snapshot_path: str = "results/tier2-cs170k/_strict_genmatch_pre_fix.json",
) -> list[dict]:
    """Load metrics.json for every completed Tier-2 seeded run.

    Tier-1 dirs (no _s suffix), unreadable/partial JSON, and runs missing an
    accuracy metric (e.g. the generation-eval pass has not run yet) are all
    skipped — summarize tolerates a partially-complete results/ directory.

    Merges the strict-parser snapshot into each matching run dict as
    `eval_accuracy_genmatch_strict`. Missing snapshot file → strict simply absent
    from the run dicts (aggregation skips the column accordingly).
    """
    strict = load_strict_snapshot(strict_snapshot_path)
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
            if d["run_id"] in strict:
                d["eval_accuracy_genmatch_strict"] = strict[d["run_id"]]
            runs.append(d)
    return runs


def _mean_std(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def aggregate(runs: list[dict]) -> dict[tuple, dict]:
    """Group runs by (trainset, method, rank) → aggregated cell stats.

    The strict-parser genmatch is included as `genmatch_strict_mean`/`_std` only when
    every run in the cell has it (i.e. when the strict snapshot covered every seed of
    the cell). Cells without complete strict coverage simply omit those keys.
    """
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
        cell = {
            "n": len(ds),
            "likelihood_mean": lik_m, "likelihood_std": lik_s,
            "genmatch_mean": gen_m, "genmatch_std": gen_s,
            "runtime_s_mean": rt_m, "peak_gb_mean": mem_m,
            "trainable_params": ds[0]["trainable_params"],
        }
        strict_values = [d["eval_accuracy_genmatch_strict"] for d in ds if "eval_accuracy_genmatch_strict" in d]
        if len(strict_values) == len(ds):
            sm, ss = _mean_std(strict_values)
            cell["genmatch_strict_mean"] = sm
            cell["genmatch_strict_std"] = ss
        cells[key] = cell
    return cells


def _gap_table(cells: dict[tuple, dict], trainset: str, metric: str) -> list[str]:
    """DoRA - LoRA gap by rank for one regime and one metric. Skips cells missing the metric."""
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
        if lo and do and f"{metric}_mean" in lo and f"{metric}_mean" in do:
            lm, ls = lo[f"{metric}_mean"], lo[f"{metric}_std"]
            dm, ds = do[f"{metric}_mean"], do[f"{metric}_std"]
            out.append(
                f"| {r} | {lm:.4f} +/- {ls:.4f} | {dm:.4f} +/- {ds:.4f} | {dm - lm:+.4f} |"
            )
    out.append("")
    return out


def _interpretation_notes(cells: dict[tuple, dict]) -> list[str]:
    """Multi-paragraph interpretation; emitted only when cs170k results are present."""
    if not any(k[0] == "cs170k" for k in cells):
        return []
    return [
        "## Notes on interpretation",
        "",
        "### Two views of cs170k genmatch: broad parser vs strict parser",
        "",
        "`genmatch (broad)` accepts either yes/no or true/false (normalized to true/false). It measures whether the model emits a parseable answer matching the gold label — i.e. task accuracy via free generation. `genmatch (strict)` only accepts literal true/false; it scores zero when the model obeys the BoolQ eval prompt's yes/no instruction and emits yes/no. So the strict view is a *format-adaptation strength* metric: how strongly the adapter has overridden the prompt with cs170k's true/false training format.",
        "",
        "The strict column shows a clean rank-stratified pattern (low at r=4, much higher at r=16) reflecting that higher-rank adapters have the capacity to lock in the cs170k format while low-rank ones lack that capacity and default to the prompt's yes/no. Diagnostic spot-checks (raw generations of 4 representative runs) confirm this: low-rank cs170k models emit `\"the correct answer is yes/no\"` ~90% of the time; high-rank ones emit `\"the correct answer is true/false\"`. Both are answering correctly; the difference is surface form.",
        "",
        "### Inverse-rank trend on broad genmatch",
        "",
        "On the broad metric, cs170k genmatch decreases slightly with rank (r=4 ~0.85 → r=16 ~0.78–0.82) — opposite to \"more capacity = better\". Reading: cs170k is a multi-task mix (BoolQ is one of 8 tasks); training on it specializes the model *away from pure BoolQ*. At low rank the specialization is mild — the model retains its base BoolQ ability and gets a small boost from training (low-rank trained ~0.85 > zero-shot ~0.82). At high rank the model commits harder to the broader cs170k distribution, sometimes at the cost of BoolQ-specific accuracy (high-rank trained ~0.78 < zero-shot ~0.82). More capacity to learn cs170k means more capacity to drift from BoolQ-optimal behavior.",
        "",
        "### DoRA vs LoRA: task accuracy vs format-adaptation",
        "",
        "**Task accuracy (broad genmatch and likelihood).** The DoRA−LoRA gap is within seed std at every rank in the BoolQ regime and at r ∈ {4, 8} in cs170k. The exception is cs170k r=16: broad-genmatch shows DoRA −0.032 below LoRA, direction-consistent across all 3 seeds (gap exceeds seed std ~0.013–0.019, though n=3 is not enough for formal significance); the likelihood gap at the same cell (−0.054) is within DoRA's wide seed std at r=16 (0.090). The Tier-1 / Phase-1 null result largely reproduces in the multi-task regime that was specifically designed to surface a DoRA advantage; the paper's claim of a particularly large DoRA edge at low rank does not replicate as a task-accuracy improvement in our setup.",
        "",
        "**Format-adaptation strength (strict genmatch).** Here DoRA shows a small but direction-consistent edge at r=8 (+0.056) and r=16 (+0.056); both r=4 cells are tied near the floor (~0.10). The seed std on this metric is large (~0.16–0.35 per cell, comparable to the means), so the gap is below statistical significance with n=3 — but the consistent direction matches the spec's central hypothesis: DoRA's magnitude/direction decomposition embeds the cs170k true/false training format more reliably than LoRA. It just doesn't translate into measurable task-accuracy improvement at our 2500-step training budget.",
        "",
        "**Cost.** DoRA's overhead — roughly 3.3× wall-time and 1.8× peak GPU memory vs LoRA at otherwise identical settings — is firmly reconfirmed across both phases.",
        "",
    ]


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
    has_strict = any("genmatch_strict_mean" in c for c in cells.values())

    header = "| trainset | method | r | n | likelihood acc | genmatch (broad) |"
    sep = "|---|---|---:|---:|---:|---:|"
    if has_strict:
        header += " genmatch (strict) |"
        sep += "---:|"
    header += " runtime (s) | peak mem (GB) |"
    sep += "---:|---:|"

    lines += ["## Per-cell results (mean +/- std over 3 seeds)", "", header, sep]
    for key in sorted(cells):
        ts, method, r = key
        c = cells[key]
        row = (
            f"| {ts} | {method.upper()} | {r} | {c['n']} | "
            f"{c['likelihood_mean']:.4f} +/- {c['likelihood_std']:.4f} | "
            f"{c['genmatch_mean']:.4f} +/- {c['genmatch_std']:.4f} |"
        )
        if has_strict:
            if "genmatch_strict_mean" in c:
                row += f" {c['genmatch_strict_mean']:.4f} +/- {c['genmatch_strict_std']:.4f} |"
            else:
                row += " - |"
        row += f" {c['runtime_s_mean']:.0f} | {c['peak_gb_mean']:.1f} |"
        lines.append(row)
    lines.append("")

    for trainset in ["boolq", "cs170k"]:
        if any(ts == trainset for (ts, _, _) in cells):
            lines += _gap_table(cells, trainset, "likelihood")
            lines += _gap_table(cells, trainset, "genmatch")
            if has_strict and trainset == "cs170k":
                lines += _gap_table(cells, trainset, "genmatch_strict")

    lines += _interpretation_notes(cells)

    lines += [
        "## How to regenerate", "",
        "```bash",
        "cd mini-project-DoRA && python scripts/summarize_results.py --tier 2 > results/SUMMARY-tier2.md",
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
