"""Regenerate the three on-slide figures as SVG for the oral website.

Standalone — does not touch the canonical PNG-producing notebooks
(`notebooks/analysis.py`, `notebooks/analysis_tier3.py`) or the canonical
PNG outputs under `figures/`. The matplotlib code below mirrors those
notebooks (same data, same styling) but writes SVG into `oral/figures/`.

Run from the project's local venv:

    .venv/bin/python oral/gen_svgs.py

Three SVGs produced:
- oral/figures/rank_sensitivity.svg     (T2 — left+right panel, BoolQ+cs170k)
- oral/figures/format_adaptation.svg    (T2 — strict vs broad parser by rank)
- oral/figures/tier3/rank_sensitivity.svg  (T3 — headline divergence figure)

The three currently-unreferenced PNGs (loss_curves variants, format_drift) are
left as PNG and not regenerated as SVG — if the slide deck adds them later,
extend this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dora_mini import summarize, summarize_tier3  # noqa: E402

OUT_T2 = HERE / "figures"
OUT_T3 = HERE / "figures" / "tier3"
OUT_T2.mkdir(parents=True, exist_ok=True)
OUT_T3.mkdir(parents=True, exist_ok=True)

# Shared style — keeps cross-tier figures visually consistent.
RANKS = [4, 8, 16]
METHOD_COLORS = {"lora": "C0", "dora": "C1"}
METRIC_STYLES = {"likelihood": "-", "genmatch": "--"}
ZEROSHOT_GEN = 0.82

# Bigger fonts so they're readable when projected without zooming.
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    # Embed text as text (the SVG point of selling it) rather than paths.
    "svg.fonttype": "none",
})


def load_tier2_cells():
    runs = []
    for sub in ("tier1", "tier2-boolq", "tier2-cs170k", "tier2-baseline"):
        runs += summarize.load_runs(
            pattern=str(PROJECT_ROOT / "results" / sub / "**" / "metrics.json"),
            strict_snapshot_path=str(
                PROJECT_ROOT / "results" / "tier2-cs170k" / "_strict_genmatch_pre_fix.json"
            ),
        )
    cells = summarize.aggregate(runs)
    print(f"T2: {len(runs)} runs, {len(cells)} cells")
    return cells


def load_tier3_cells():
    runs = summarize_tier3.load_runs(
        pattern=str(PROJECT_ROOT / "results" / "tier3-cs170k" / "*" / "metrics.json"),
    )
    cells = summarize_tier3.aggregate(runs)
    print(f"T3: {len(runs)} runs, {len(cells)} cells")
    return cells


def fig_t2_rank_sensitivity(cells) -> None:
    """Two-panel: BoolQ (left), cs170k (right), with the zero-shot reference."""
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
            ax.axhline(
                ZEROSHOT_GEN, color="grey", linewidth=0.8,
                linestyle="-.", label=f"zero-shot (~{ZEROSHOT_GEN:.2f})",
            )
        ax.set_title(f"trained on {trainset}")
        ax.set_xlabel("rank r")
        ax.set_xticks(RANKS)
    axes[0].set_ylabel("BoolQ dev accuracy")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    out = OUT_T2 / "rank_sensitivity.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")


def fig_t2_format_adaptation(cells) -> None:
    """cs170k strict vs broad parser by rank — the parser-fix story."""
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
    ax.axhline(
        ZEROSHOT_GEN, color="grey", linewidth=0.8,
        linestyle="-.", label=f"zero-shot broad (~{ZEROSHOT_GEN:.2f})",
    )
    ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":", label="chance (0.5)")
    ax.set_title("cs170k: format-adaptation (strict) vs task accuracy (broad)")
    ax.set_xlabel("rank r")
    ax.set_ylabel("BoolQ dev genmatch accuracy")
    ax.set_xticks(RANKS)
    ax.set_ylim(-0.05, 1.0)
    # Bottom-right corner sits below all six data lines at r=16
    # (strict tops out at ~0.63; chance ref at 0.5) — no overlap.
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = OUT_T2 / "format_adaptation.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")


def fig_t3_rank_sensitivity(cells) -> None:
    """T3 headline — likelihood collapse vs genmatch hold, one panel."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for method in ["lora", "dora"]:
        for metric in ["likelihood", "genmatch"]:
            ys, es = [], []
            for r in RANKS:
                c = cells.get(("cs170k", method, r))
                ys.append(c[f"{metric}_mean"] if c else float("nan"))
                es.append(c[f"{metric}_std"] if c else 0.0)
            ax.errorbar(
                RANKS, ys, yerr=es,
                marker="o", capsize=3,
                color=METHOD_COLORS[method],
                linestyle=METRIC_STYLES[metric],
                label=f"{method.upper()} ({metric})",
            )
    ax.axhline(
        ZEROSHOT_GEN, color="grey", linewidth=0.8,
        linestyle="-.", label=f"zero-shot ({ZEROSHOT_GEN:.2f})",
    )
    ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":", label="chance (0.5)")
    ax.set_title("Tier 3 cs170k (10k steps, n=4): rank sensitivity")
    ax.set_xlabel("rank r")
    ax.set_ylabel("BoolQ dev accuracy")
    ax.set_xticks(RANKS)
    ax.set_ylim(0.3, 0.95)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = OUT_T3 / "rank_sensitivity.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")


def main() -> None:
    t2 = load_tier2_cells()
    t3 = load_tier3_cells()
    fig_t2_rank_sensitivity(t2)
    fig_t2_format_adaptation(t2)
    fig_t3_rank_sensitivity(t3)
    print("done.")


if __name__ == "__main__":
    main()
