# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

# %% [markdown]
# # Tier 3 cs170k enrichment — figures
#
# Produces the three Tier-3 deliverable figures under `figures/tier3/`:
# 1. `rank_sensitivity.png` — likelihood vs genmatch per cell, by rank, with
#    zero-shot reference line. The headline figure: visualizes the inverse-rank
#    trend on genmatch + the format-vs-task divergence in one panel.
# 2. `format_drift.png` — genmatch minus likelihood gap per cell by rank, per
#    method. Visualizes how the format-preservation gap grows with rank.
# 3. `loss_curves.png` — train and eval loss for a representative T3 cell
#    (`dora_r8_s114_t3` — middle-of-distribution on every axis).
#
# **Scope: Tier-3 only.** Outputs go to `figures/tier3/`, never to `figures/` directly.
# The three canonical Tier-2 PNGs at `figures/{rank_sensitivity,format_adaptation,loss_curves}.png`
# are produced by the sibling notebook `analysis.py` and are not touched here.
# Loads via `summarize_tier3.aggregate`; pulls Tier-2 cs170k cells via the
# unmodified `summarize.aggregate` for overlay where useful.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd().parent / "src"))
from dora_mini import summarize, summarize_tier3  # noqa: E402

FIGDIR = Path.cwd().parent / "figures" / "tier3"
FIGDIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path.cwd().parent
t3_runs = summarize_tier3.load_runs(
    pattern=str(PROJECT_ROOT / "results" / "tier3-cs170k" / "*" / "metrics.json"),
)
t3_cells = summarize_tier3.aggregate(t3_runs)
t2_runs = []
for sub in ("tier2-cs170k",):
    t2_runs += summarize.load_runs(
        pattern=str(PROJECT_ROOT / "results" / sub / "**" / "metrics.json"),
        strict_snapshot_path=str(PROJECT_ROOT / "results" / "tier2-cs170k" / "_strict_genmatch_pre_fix.json"),
    )
t2_cells = summarize.aggregate(t2_runs)
print(f"T3: {len(t3_runs)} runs, {len(t3_cells)} cells")
print(f"T2 cs170k: {len(t2_runs)} runs, {len(t2_cells)} cells")

# Style convention — matches T2's analysis.py for cross-doc consistency.
RANKS = [4, 8, 16]
METHOD_COLORS = {"lora": "C0", "dora": "C1"}
METRIC_STYLES = {"likelihood": "-", "genmatch": "--"}
ZEROSHOT_GEN = 0.82  # Mistral-7B-Instruct zero-shot broad-genmatch on BoolQ dev (from tier2-baseline)

# %% [markdown]
# ## Figure 1 — Tier-3 rank sensitivity (the headline)
# Both metrics on the same panel so the likelihood-collapse vs genmatch-holds divergence is visually obvious.
# Zero-shot baseline drawn for reference; the inverse-rank trend on genmatch lands neatly above the zero-shot
# line at r=4 and clearly below it at r=16.

# %%
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for method in ["lora", "dora"]:
    for metric in ["likelihood", "genmatch"]:
        ys, es = [], []
        for r in RANKS:
            c = t3_cells.get(("cs170k", method, r))
            ys.append(c[f"{metric}_mean"] if c else float("nan"))
            es.append(c[f"{metric}_std"] if c else 0.0)
        ax.errorbar(
            RANKS, ys, yerr=es,
            marker="o", capsize=3,
            color=METHOD_COLORS[method],
            linestyle=METRIC_STYLES[metric],
            label=f"{method.upper()} ({metric})",
        )
ax.axhline(ZEROSHOT_GEN, color="grey", linewidth=0.8, linestyle="-.", label=f"zero-shot ({ZEROSHOT_GEN:.2f})")
ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":", label="chance (0.5)")
ax.set_title("Tier 3 cs170k (10k steps, n=4): rank sensitivity")
ax.set_xlabel("rank r")
ax.set_ylabel("BoolQ dev accuracy")
ax.set_xticks(RANKS)
ax.set_ylim(0.3, 0.95)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(FIGDIR / "rank_sensitivity.png", dpi=150)

# %% [markdown]
# ## Figure 2 — format-drift gap by rank
# `genmatch - likelihood` per cell. Visualizes how the format-preservation gap (i.e. how much "harder" the
# model is making it for the yes/no probe vs the broad-vocabulary probe) grows with rank. A flat curve at zero
# would mean the model still answers in yes/no; a positive curve means the model has switched format.

# %%
fig, ax = plt.subplots(figsize=(6.5, 4.0))
for method in ["lora", "dora"]:
    ys = []
    for r in RANKS:
        c = t3_cells.get(("cs170k", method, r))
        if c and "genmatch_mean" in c:
            ys.append(c["genmatch_mean"] - c["likelihood_mean"])
        else:
            ys.append(float("nan"))
    ax.plot(
        RANKS, ys,
        marker="o",
        color=METHOD_COLORS[method],
        linestyle="-",
        label=f"{method.upper()}",
    )
ax.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
ax.set_title("Tier 3 cs170k: format-drift gap (genmatch - likelihood) by rank")
ax.set_xlabel("rank r")
ax.set_ylabel("genmatch - likelihood (per cell mean)")
ax.set_xticks(RANKS)
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(FIGDIR / "format_drift.png", dpi=150)

# %% [markdown]
# ## Figure 3 — train and eval loss curves for a representative Tier-3 cell
# Picks `dora_mistral7b_cs170k_r8_s114_t3` — middle-of-distribution: mid rank, mid genmatch, no extreme seed
# behavior. The train curve shows tight convergence by ~step 1200 and continued descent to step 10k; the eval
# curve (BoolQ-yes/no monitor set, 500 examples) tells the format-shift story — flat-then-rising as the model
# drifts away from BoolQ-friendly behavior.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
target = "dora_mistral7b_cs170k_r8_s114_t3"
chosen = None
for d in t3_runs:
    if d["run_id"] == target:
        chosen = d
        break
if chosen is None:
    raise RuntimeError(f"representative run {target} not found in t3_runs")
steps = [p["step"] for p in chosen["loss_curve"]]
ax.plot(steps, [p["loss"] for p in chosen["loss_curve"]], label="train loss")
eval_pts = chosen.get("eval_loss_curve", [])
if eval_pts:
    ax.plot(
        [p["step"] for p in eval_pts],
        [p["eval_loss"] for p in eval_pts],
        "--", label="BoolQ-yes/no monitor eval loss",
    )
ax.set_xlabel("step")
ax.set_ylabel("loss (log scale)")
ax.set_yscale("log")
ax.set_title(f"Tier 3 train + eval loss: {target}")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIGDIR / "loss_curves.png", dpi=150)

# %% [markdown]
# ## Done
# Three figures written to `figures/tier3/`. See `results/SUMMARY-tier3.md` for the numerical tables and
# interpretation prose that these figures support.

# %%
print("wrote figures/tier3/:")
for f in sorted(FIGDIR.glob("*.png")):
    print(f"  {f.name} ({f.stat().st_size // 1024} KB)")
