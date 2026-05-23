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

PROJECT_ROOT = Path.cwd().parent
runs = summarize.load_runs(
    pattern=str(PROJECT_ROOT / "results" / "**" / "metrics.json"),
    strict_snapshot_path=str(PROJECT_ROOT / "results" / "tier2-cs170k" / "_strict_genmatch_pre_fix.json"),
)
cells = summarize.aggregate(runs)
print(f"{len(runs)} runs, {len(cells)} cells")

# %% [markdown]
# ## Headline: DoRA - LoRA gap vs rank, per regime and metric

# %%
RANKS = [4, 8, 16]
# Convention used in both rank_sensitivity and format_adaptation: color encodes
# method (LoRA blue, DoRA orange); linestyle encodes the metric variant.
METHOD_COLORS = {"lora": "C0", "dora": "C1"}
METRIC_STYLES = {"likelihood": "-", "genmatch": "--"}
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
                label=f"{method.upper()} ({metric})",
            )
    if trainset == "cs170k":
        ax.axhline(0.82, color="grey", linewidth=0.8, linestyle="-.", label="zero-shot (~0.82)")
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
            ("genmatch", "broad", "-"),
            ("genmatch_strict", "strict", "--"),
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
    ax.axhline(0.82, color="grey", linewidth=0.8, linestyle="-.", label="zero-shot broad (~0.82)")
    ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":", label="chance (0.5)")
    ax.set_title("cs170k: format-adaptation (strict) vs task accuracy (broad)")
    ax.set_xlabel("rank r")
    ax.set_ylabel("BoolQ dev genmatch accuracy")
    ax.set_xticks(RANKS)
    ax.set_ylim(-0.05, 1.0)
    ax.legend(fontsize=9, loc="center right")
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
