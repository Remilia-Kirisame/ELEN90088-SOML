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
# # Tier 2 DoRA Regime Study — analysis
# Rank-sensitivity (per regime, per metric), overfitting curves, and cost.
#
# **Scope: Tier-2 only.** This notebook generates the three canonical Tier-2 figures
# under `figures/` (`rank_sensitivity.png`, `format_adaptation.png`, `loss_curves.png`).
# Tier-3 figures live under `figures/tier3/` and are produced by a separate
# notebook (`analysis_tier3.py`); this file's glob is intentionally narrowed to
# `results/tier2-*/` so that a stray re-run of *this* notebook can never pollute
# the Tier-2 deliverable PNGs with Tier-3 data. Even though `summarize._RUN_ID`
# already rejects Tier-3's `_t3`-suffixed ids by regex, making the scope explicit
# at the glob level removes the implicit safety dependency.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd().parent / "src"))
from dora_mini import summarize  # noqa: E402

FIGDIR = Path.cwd().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

PROJECT_ROOT = Path.cwd().parent
# Tier-2 only — Tier-3 is loaded by a sibling notebook. See module-level docstring above.
runs = []
for sub in ("tier1", "tier2-boolq", "tier2-cs170k", "tier2-baseline"):
    runs += summarize.load_runs(
        pattern=str(PROJECT_ROOT / "results" / sub / "**" / "metrics.json"),
        strict_snapshot_path=str(PROJECT_ROOT / "results" / "tier2-cs170k" / "_strict_genmatch_pre_fix.json"),
    )
cells = summarize.aggregate(runs)
print(f"{len(runs)} runs, {len(cells)} cells")

# %% [markdown]
# ## Headline: DoRA - LoRA gap vs rank, per regime and metric

# %%
RANKS = [4, 8, 16]
# Convention used in both rank_sensitivity and format_adaptation: color encodes
# method (LoRA blue, DoRA orange); linestyle encodes task-accuracy-vs-diagnostic.
# SOLID = broad-parser genmatch (the canonical task-accuracy metric). DASHED =
# likelihood OR strict-parser genmatch (parser/format diagnostics that diverge
# from task accuracy when the model's output format shifts).
METHOD_COLORS = {"lora": "C0", "dora": "C1"}
METRIC_STYLES = {"likelihood": "--", "genmatch": "-"}
METRIC_LABELS = {"likelihood": "likelihood", "genmatch": "broad genmatch"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, trainset in zip(axes, ["boolq", "cs170k"]):
    for method in ["lora", "dora"]:
        for metric in ["likelihood", "genmatch"]:
            ys, es = [], []
            for r in RANKS:
                c = cells.get((trainset, method, r))
                ys.append(c[f"{metric}_mean"] if c else float("nan"))
                es.append(c[f"{metric}_std"] if c else 0.0)
            ax.errorbar(
                RANKS, ys, yerr=es,
                marker="o", capsize=3,
                color=METHOD_COLORS[method],
                linestyle=METRIC_STYLES[metric],
                label=f"{method.upper()} ({METRIC_LABELS[metric]})",
            )
    if trainset == "cs170k":
        ax.axhline(0.82, color="grey", linewidth=0.8, linestyle="-.", label="zero-shot broad-genmatch (~0.82)")
    ax.set_title(f"trained on {trainset}")
    ax.set_xlabel("rank r")
    ax.set_xticks(RANKS)
axes[0].set_ylabel("BoolQ dev accuracy")
axes[0].legend(fontsize=8, loc="lower right")
axes[1].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIGDIR / "rank_sensitivity.png", dpi=150)

# %% [markdown]
# ## Format adaptation (cs170k only): strict vs broad parser by rank
# Strict-parser genmatch only counts literal true/false → measures how strongly
# the adapter overrode the BoolQ prompt's yes/no instruction with cs170k's
# training format. Broad parser accepts either vocabulary → task accuracy.

# %%
if any(k[0] == "cs170k" and "genmatch_strict_mean" in cells[k] for k in cells):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method in ["lora", "dora"]:
        for variant, label_suffix, style in [
            ("genmatch", "broad genmatch", "-"),
            ("genmatch_strict", "strict genmatch", "--"),
        ]:
            ys, es = [], []
            for r in RANKS:
                c = cells.get(("cs170k", method, r))
                if c and f"{variant}_mean" in c:
                    ys.append(c[f"{variant}_mean"])
                    es.append(c[f"{variant}_std"])
                else:
                    ys.append(float("nan"))
                    es.append(0.0)
            ax.errorbar(
                RANKS, ys, yerr=es,
                marker="o", capsize=3,
                color=METHOD_COLORS[method],
                linestyle=style,
                label=f"{method.upper()} ({label_suffix})",
            )
    ax.axhline(0.82, color="grey", linewidth=0.8, linestyle="-.", label="zero-shot broad-genmatch (~0.82)")
    ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":", label="chance (0.5)")
    ax.set_title("cs170k: format-adaptation (strict) vs task accuracy (broad)")
    ax.set_xlabel("rank r")
    ax.set_ylabel("BoolQ dev genmatch accuracy")
    ax.set_xticks(RANKS)
    ax.set_ylim(-0.05, 1.0)
    # Bottom-right corner sits below all six data lines at r=16 (strict tops at
    # ~0.63; chance ref at 0.5) so the legend doesn't collide with the lines.
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "format_adaptation.png", dpi=150)

# %% [markdown]
# ## Overfitting - train vs eval loss (one representative run per regime)

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for d in runs:
    meta = summarize.parse_run_id(d["run_id"])
    if meta["method"] == "dora" and meta["r"] == 8 and meta["seed"] == 42:
        steps = [p["step"] for p in d["loss_curve"]]
        ax.plot(steps, [p["loss"] for p in d["loss_curve"]],
                label=f"{meta['trainset']} train")
        e = d.get("eval_loss_curve", [])
        if e:
            ax.plot([p["step"] for p in e], [p["eval_loss"] for p in e],
                    "--", label=f"{meta['trainset']} eval")
ax.set_xlabel("step")
ax.set_ylabel("loss (log scale)")
ax.set_yscale("log")
ax.set_title("Train and eval loss (representative DoRA r=8 seed=42)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "loss_curves.png", dpi=150)
