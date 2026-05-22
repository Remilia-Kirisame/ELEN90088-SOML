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

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd().parent / "src"))
from dora_mini import summarize  # noqa: E402

FIGDIR = Path.cwd().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

runs = summarize.load_runs(str(Path.cwd().parent / "results" / "**" / "metrics.json"))
cells = summarize.aggregate(runs)
print(f"{len(runs)} runs, {len(cells)} cells")

# %% [markdown]
# ## Headline: DoRA - LoRA gap vs rank, per regime and metric

# %%
RANKS = [4, 8, 16]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, trainset in zip(axes, ["boolq", "cs170k"]):
    for metric in ["likelihood", "genmatch"]:
        for method in ["lora", "dora"]:
            ys, es = [], []
            for r in RANKS:
                c = cells.get((trainset, method, r))
                ys.append(c[f"{metric}_mean"] if c else float("nan"))
                es.append(c[f"{metric}_std"] if c else 0.0)
            ax.errorbar(RANKS, ys, yerr=es, marker="o", capsize=3,
                        label=f"{method.upper()} ({metric})")
    ax.set_title(f"trained on {trainset}")
    ax.set_xlabel("rank r")
    ax.set_xticks(RANKS)
axes[0].set_ylabel("BoolQ dev accuracy")
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "rank_sensitivity.png", dpi=150)

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
ax.set_ylabel("loss")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "loss_curves.png", dpi=150)
