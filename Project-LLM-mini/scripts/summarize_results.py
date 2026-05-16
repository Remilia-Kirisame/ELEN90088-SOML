#!/usr/bin/env python3
"""Aggregate Project-LLM-mini run results into a single markdown summary.

Scans `results/*_*/metrics.json` and prints a comprehensive markdown report covering: per-run row table (config + metrics + slurm + timestamps), DoRA-vs-LoRA accuracy gaps by rank, DoRA/LoRA cost ratios, per-step training loss curves, shared configuration, environment, and a statistical-noise-floor caveat for single-seed runs.

Run from the Project-LLM-mini root:

    python scripts/summarize_results.py > results/SUMMARY.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from glob import glob


def fmt_dt(iso: str) -> str:
    """Convert an ISO-8601 string (with trailing Z or +00:00) to 'YYYY-MM-DD HH:MM UTC'."""
    if not iso:
        return ""
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def load_runs(pattern: str = "results/*_*/metrics.json") -> list[dict]:
    """Load all metrics.json files matching the glob; sort by (method, rank)."""
    files = sorted(glob(pattern))
    runs = [json.load(open(f)) for f in files]
    runs.sort(key=lambda d: (d["config_resolved"]["peft"]["method"], d["config_resolved"]["peft"]["r"]))
    return runs


def render(runs: list[dict]) -> str:
    lines: list[str] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines.append("# Tier 1 results summary")
    lines.append("")
    lines.append(f"Generated {today} (UTC) from `results/*_*/metrics.json` by `scripts/summarize_results.py`. {len(runs)} runs: LoRA vs DoRA at r ∈ {{4, 8, 16}} on BoolQ + Mistral-7B-Instruct-v0.3.")
    lines.append("")

    # --- Per-run table ---
    lines.append("## Per-run table")
    lines.append("")
    lines.append("| run_id | method | r | α | α/r | trainable | % of full | accuracy | NLL (yes/no) | runtime (s) | runtime (min) | peak mem (GB) | slurm job | started (UTC) | finished (UTC) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for d in runs:
        peft = d["config_resolved"]["peft"]
        method = peft["method"].upper()
        r, alpha = peft["r"], peft["alpha"]
        ratio = alpha / r
        rt = d["train_runtime_s"]
        lines.append(
            f"| `{d['run_id']}` | {method} | {r} | {alpha} | {ratio:.0f} | "
            f"{d['trainable_params']:,} | {d['trainable_pct']:.3f}% | "
            f"{d['eval_accuracy']:.4f} | {d['eval_loss']:.4f} | "
            f"{rt} | {rt/60:.1f} | {d['peak_memory_gb']:.1f} | "
            f"`{d.get('slurm_job_id', '')}` | {fmt_dt(d.get('started_at', ''))} | {fmt_dt(d.get('finished_at', ''))} |"
        )
    lines.append("")

    # --- DoRA-LoRA gap by rank ---
    lines.append("## DoRA − LoRA accuracy gap by rank")
    lines.append("")
    by_rank: dict[int, dict[str, float]] = {}
    for d in runs:
        peft = d["config_resolved"]["peft"]
        by_rank.setdefault(peft["r"], {})[peft["method"]] = d["eval_accuracy"]
    lines.append("| r | LoRA accuracy | DoRA accuracy | Δ (DoRA − LoRA) |")
    lines.append("|---:|---:|---:|---:|")
    for r in sorted(by_rank):
        slot = by_rank[r]
        if "lora" in slot and "dora" in slot:
            lines.append(f"| {r} | {slot['lora']:.4f} | {slot['dora']:.4f} | {slot['dora'] - slot['lora']:+.4f} |")
    lines.append("")

    # --- Cost ratios ---
    lines.append("## Cost ratios (DoRA vs LoRA at the same r)")
    lines.append("")
    cost: dict[int, dict[str, dict]] = {}
    for d in runs:
        peft = d["config_resolved"]["peft"]
        cost.setdefault(peft["r"], {})[peft["method"]] = {
            "runtime_s": d["train_runtime_s"],
            "peak_gb": d["peak_memory_gb"],
            "params": d["trainable_params"],
        }
    lines.append("| r | runtime ratio | peak mem ratio | param overhead (DoRA − LoRA) |")
    lines.append("|---:|---:|---:|---:|")
    for r in sorted(cost):
        slot = cost[r]
        if "lora" in slot and "dora" in slot:
            l, do = slot["lora"], slot["dora"]
            lines.append(
                f"| {r} | {do['runtime_s'] / l['runtime_s']:.2f}× | "
                f"{do['peak_gb'] / l['peak_gb']:.2f}× | "
                f"+{do['params'] - l['params']:,} |"
            )
    lines.append("")

    # --- Per-step train loss ---
    lines.append("## Per-step training loss")
    lines.append("")
    all_steps = sorted({s["step"] for d in runs for s in d["loss_curve"]})
    header = "| run_id | " + " | ".join(f"step {s}" for s in all_steps) + " |"
    sep = "|---|" + "---:|" * len(all_steps)
    lines.append(header)
    lines.append(sep)
    for d in runs:
        curve = {s["step"]: s["loss"] for s in d["loss_curve"]}
        cells = [f"{curve[s]:.4f}" if s in curve else "" for s in all_steps]
        lines.append(f"| `{d['run_id']}` | " + " | ".join(cells) + " |")
    lines.append("")

    # --- Shared config ---
    lines.append("## Shared configuration")
    lines.append("")
    cfg = runs[0]["config_resolved"]
    lines.append(f"- **Model:** `{cfg['model']['name']}` ({cfg['model']['dtype']})")
    lines.append(f"- **Dataset:** `{cfg['data']['dataset']}` (train_size={cfg['data']['train_size']}, eval_size={cfg['data']['eval_size']}, max_length={cfg['data']['max_length']})")
    lines.append(f"- **PEFT:** `target_modules={cfg['peft']['target_modules']}`, `dropout={cfg['peft']['dropout']}` (r and α vary per run)")
    eff_batch = cfg["training"]["batch_size"] * cfg["training"]["grad_accum"]
    lines.append(f"- **Training:** batch_size={cfg['training']['batch_size']} × grad_accum={cfg['training']['grad_accum']} (effective batch = {eff_batch}), lr={cfg['training']['learning_rate']}, num_steps={cfg['training']['num_steps']}, warmup_steps={cfg['training']['warmup_steps']}, seed={cfg['training']['seed']}")
    lines.append("")

    # --- Environment ---
    lines.append("## Environment")
    lines.append("")
    env = runs[0]["env"]
    lines.append(f"- Python {env['python']} · torch {env['torch']} · transformers {env['transformers']} · peft {env['peft']}")
    lines.append(f"- GPU: {env['gpu']} (Spartan `gpu-h100` partition)")
    lines.append("")

    # --- Noise floor ---
    lines.append("## Statistical caveat")
    lines.append("")
    lines.append("Single seed per config, `eval_size=500`. Binomial standard error on a single accuracy measurement at $p \\approx 0.88$:")
    lines.append("")
    lines.append("$$")
    lines.append("\\mathrm{SE} = \\sqrt{\\frac{p(1-p)}{n}} = \\sqrt{\\frac{0.88 \\cdot 0.12}{500}} \\approx 0.0145")
    lines.append("$$")
    lines.append("")
    lines.append("So per-measurement SE ≈ ±1.45 pp at 1σ, ±2.9 pp at 2σ. Every observed LoRA-vs-DoRA gap in our table (≤ 0.6 pp) is well inside one SE — cannot be statistically distinguished from zero without multiple seeds.")
    lines.append("")

    # --- Regen note ---
    lines.append("## How to regenerate")
    lines.append("")
    lines.append("```bash")
    lines.append("cd Project-LLM-mini && python scripts/summarize_results.py > results/SUMMARY.md")
    lines.append("```")
    lines.append("")
    lines.append("Re-run after adding new entries under `results/<run_id>/metrics.json`; the script picks up new runs by glob.")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    runs = load_runs()
    if not runs:
        sys.stderr.write("no metrics.json found under results/\n")
        return 1
    print(render(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
