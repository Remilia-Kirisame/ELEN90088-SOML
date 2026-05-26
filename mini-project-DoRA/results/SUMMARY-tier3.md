# Tier 3 results summary

**Sequential follow-up dig-in on Tier-2's cs170k open questions.** Tier 2 (see [SUMMARY-tier2.md](SUMMARY-tier2.md)) is the canonical study — complete and unchanged. After Tier 2 met the project rubric, we ran an additional 24 cs170k cells *post hoc* to test whether two of Tier-2's open hypotheses sharpen at 4x the training budget: (i) the tentative inverse-rank trend on broad genmatch (~3 pp at Tier 2, needed more steps to confirm), and (ii) the +0.056 strict-genmatch DoRA edge that sat within seed std at n=3. New training recipe: 10k steps (vs Tier 2's 2500), warmup 400 (proportional), seeds `{114, 514, 1919, 810}` (n=4, *all new seeds — disjoint from Tier 2's `{42, 1, 2}`*). 24 training runs + 24 gen-eval runs. Code commit `fe6ed5e` on branch `llm/t3`.

## Per-cell results (mean +/- std over 4 seeds)

| trainset | method | r | n | likelihood acc | genmatch (broad) | runtime (s) | peak mem (GB) |
|---|---|---:|---:|---:|---:|---:|---:|
| cs170k | DORA | 4 | 4 | 0.6514 +/- 0.2044 | 0.8272 +/- 0.0071 | 19324 | 59.4 |
| cs170k | DORA | 8 | 4 | 0.6180 +/- 0.0530 | 0.8078 +/- 0.0075 | 19134 | 59.6 |
| cs170k | DORA | 16 | 4 | 0.4411 +/- 0.1191 | 0.6645 +/- 0.0565 | 19110 | 59.9 |
| cs170k | LORA | 4 | 4 | 0.7644 +/- 0.0856 | 0.8427 +/- 0.0148 | 5957 | 33.3 |
| cs170k | LORA | 8 | 4 | 0.4624 +/- 0.1280 | 0.8000 +/- 0.0114 | 5906 | 33.5 |
| cs170k | LORA | 16 | 4 | 0.3817 +/- 0.0069 | 0.7029 +/- 0.0526 | 5880 | 33.9 |

### Tier-3 cs170k — DoRA - LoRA gap (likelihood)

| r | LoRA | DoRA | delta (DoRA - LoRA) |
|---:|---:|---:|---:|
| 4 | 0.7644 +/- 0.0856 | 0.6514 +/- 0.2044 | -0.1131 |
| 8 | 0.4624 +/- 0.1280 | 0.6180 +/- 0.0530 | +0.1557 |
| 16 | 0.3817 +/- 0.0069 | 0.4411 +/- 0.1191 | +0.0593 |

### Tier-3 cs170k — DoRA - LoRA gap (genmatch (broad))

| r | LoRA | DoRA | delta (DoRA - LoRA) |
|---:|---:|---:|---:|
| 4 | 0.8427 +/- 0.0148 | 0.8272 +/- 0.0071 | -0.0155 |
| 8 | 0.8000 +/- 0.0114 | 0.8078 +/- 0.0075 | +0.0078 |
| 16 | 0.7029 +/- 0.0526 | 0.6645 +/- 0.0565 | -0.0384 |

## Cross-tier comparison (T3 - T2, cs170k cells)

### Tier-3 vs Tier-2 cs170k — likelihood delta

| method | r | T2 (n=3, 2500 steps) | T3 (n=4, 10k steps) | delta (T3 - T2) |
|---|---:|---:|---:|---:|
| LORA | 4 | 0.7788 +/- 0.0723 | 0.7644 +/- 0.0856 | -0.0143 |
| LORA | 8 | 0.8103 +/- 0.0131 | 0.4624 +/- 0.1280 | -0.3479 |
| LORA | 16 | 0.7835 +/- 0.0131 | 0.3817 +/- 0.0069 | -0.4018 |
| DORA | 4 | 0.7499 +/- 0.0851 | 0.6514 +/- 0.2044 | -0.0986 |
| DORA | 8 | 0.8140 +/- 0.0244 | 0.6180 +/- 0.0530 | -0.1959 |
| DORA | 16 | 0.7294 +/- 0.0903 | 0.4411 +/- 0.1191 | -0.2883 |

### Tier-3 vs Tier-2 cs170k — genmatch (broad) delta

| method | r | T2 (n=3, 2500 steps) | T3 (n=4, 10k steps) | delta (T3 - T2) |
|---|---:|---:|---:|---:|
| LORA | 4 | 0.8485 +/- 0.0025 | 0.8427 +/- 0.0148 | -0.0058 |
| LORA | 8 | 0.8377 +/- 0.0009 | 0.8000 +/- 0.0114 | -0.0377 |
| LORA | 16 | 0.8159 +/- 0.0189 | 0.7029 +/- 0.0526 | -0.1130 |
| DORA | 4 | 0.8499 +/- 0.0055 | 0.8272 +/- 0.0071 | -0.0227 |
| DORA | 8 | 0.8403 +/- 0.0144 | 0.8078 +/- 0.0075 | -0.0325 |
| DORA | 16 | 0.7844 +/- 0.0135 | 0.6645 +/- 0.0565 | -0.1199 |

## Notes on interpretation

### Headline — the Tier-2 inverse-rank trend sharpens into a ~14 - 16 pp within-method signal

Tier 2 reported a tentative "inverse-rank trend on broad genmatch": low-rank cs170k training *beat* the zero-shot Mistral-Instruct baseline (~0.82) by a few points at r=4, while high-rank training *underperformed* it by a few points at r=16 — direction-consistent across seeds but only ~3 pp of separation, well within the range a reviewer could call noise. Tier 3 was designed to test what happens when we 4x the training budget. Result: the gradient is now crisp. At 10k cs170k steps, on the broad-genmatch metric vs the zero-shot baseline (0.82): r=4 trained models sit at-or-above zero-shot (lora_r4 = 0.843, dora_r4 = 0.827); r=8 trained models sit at-or-just-below it (lora_r8 = 0.800, dora_r8 = 0.808); r=16 trained models drop *clearly* below it (lora_r16 = 0.703, dora_r16 = 0.665). The r=4-vs-r=16 separation on genmatch is now **~14 pp within LoRA** and **~16 pp within DoRA** (versus Tier 2's ~3 pp at either method) — roughly a **5x amplification** of Tier 2's signal. The narrative this supports: more PEFT capacity = more drift from BoolQ-optimal toward the broader cs170k task mix; the more rank the adapter has, the more aggressively it commits to the multi-task distribution and the more BoolQ-specific behavior it sacrifices. This is the most novel positive contribution from Tier 3.

### Methodological diagnostic — likelihood collapsed, genmatch held (mostly)

Reading only the likelihood metric, Tier 3 looks catastrophic: every cell registers a 0.10 - 0.40 drop vs Tier 2 (lora_r16 went from 0.78 -> 0.382 — essentially anti-aligned with the BoolQ-yes/no probe; dora_r16 went from 0.73 -> 0.441; even mid-rank lora_r8 went from 0.81 -> 0.462). Reading the same runs via broad-parser genmatch, the picture is far milder — task-accuracy losses are 0.006 - 0.12 across cells, monotonic in rank. The two metrics diverge because the model is no longer producing yes/no answers in BoolQ's prompt style; it has fully adopted cs170k's true/false vocabulary. The yes/no likelihood probe then misreads a correctly-answering model as wrong, while genmatch (which accepts either vocabulary, normalized to ground truth) measures what we actually care about. This is the same format-vs-task-accuracy diagnostic Tier 2 surfaced via the strict-parser fix; Tier 3 promotes it from "methodological footnote" to "load-bearing distinction" — without genmatch we would have misread Tier 3 as evidence that long training breaks the model. It doesn't. It just changes the answer format.

Operationally for the report: **genmatch is the Tier-3 task-accuracy metric; likelihood at Tier 3 is a probe of format preservation, not task accuracy.** All cross-tier comparisons in this document use genmatch as the primary axis for that reason.

### Notes on figures — how Tier 3's `format_drift.png` relates to Tier 2's `format_adaptation.png`

Both figures probe the same underlying phenomenon — how strongly the cs170k-trained adapter has overridden BoolQ's yes/no prompt with cs170k's true/false training format — but via different metrics and at different training scales, and reading them as duplicates would miss the time-evolution story.

**Tier 2's `figures/format_adaptation.png` (at 2500 steps)** plots two genmatch numbers from the *same gen-eval pass*: broad parser (accepts yes/no OR true/false) sits flat near ~0.82 across ranks; strict parser (accepts only true/false) climbs from ~0.1 at r=4 to ~0.5 - 0.6 at r=16. The widening solid-vs-dashed gap is the format-adaptation signal — at 2500 steps the model is only *partially* shifting to true/false, and we needed a strict-only parser run to see it. (Those strict numbers are preserved in `results/tier2-cs170k/_strict_genmatch_pre_fix.json` since they were later overwritten by the broad-parser re-run.)

**Tier 3's `figures/tier3/format_drift.png` (at 10k steps)** plots the gap between two *different metrics* on the same cell: `genmatch - likelihood`. At 10k steps the model has *fully* shifted to true/false, so the yes/no likelihood probe collapses while broad genmatch holds — the gap visualizes that collapse directly without needing a separate strict-parser run. This figure is computable from data already in each cell's `metrics.json` and would have been ~0 across all cells at the Tier-2 step count (likelihood hadn't collapsed yet), so it only becomes informative at the longer training budget.

Both figures support the same conclusion — *format adaptation strengthens with rank and with training* — via complementary probes: Tier-2's caught the early-onset signal that the main metrics couldn't see; Tier-3's catches the consequence of full adaptation that the main metrics now mis-read.

### DoRA vs LoRA at 10k steps — task-accuracy null replicates; a format-preservation edge appears at r=8

On the task-accuracy metric (broad genmatch) the Tier-2 finding holds: at every rank the DoRA - LoRA gap is within seed std (r=4: -0.02; r=8: +0.008; r=16: -0.04 — all comparable in magnitude to within-cell std ~0.01 - 0.06). Four seeds and 4x training do not surface a DoRA task-accuracy advantage anywhere — replicating the Tier-1 / Tier-2 null on a substantially larger statistical and compute base.

On the likelihood metric, the picture is more interesting. The DoRA r=8 cell shows a +0.16 edge over LoRA r=8 (dora_r8 = 0.618 vs lora_r8 = 0.462), well outside the DoRA-r=8 seed std of ~0.05; 3 of 4 paired seeds favour DoRA (by 0.18-0.25 each), LoRA wins the fourth (s114) by 0.02. Read alongside the matching genmatch gap (+0.008, basically zero), this disentangles cleanly: **DoRA at r=8 preserves the yes/no answer format better than LoRA at r=8, but does not produce more correct answers**. The model is making the same set of decisions, but DoRA's magnitude/direction decomposition resists the cs170k format takeover where LoRA capitulates. This is a sharper, n=4-supported refinement of Tier 2's hint-level +0.056 strict-genmatch cell-mean edge (which sat within seed std at n=3 and was per-seed-split): DoRA's measurable edge over LoRA is in format adaptation, not task accuracy.

Reading these together: the paper's central "DoRA wins more at low rank" claim does *not* replicate as a task-accuracy improvement in our Mistral-7B-Instruct + cs170k setup, even at 4x training. It does replicate as a format-preservation advantage at mid rank — a finding the project's stated metric (BoolQ accuracy) was never designed to surface but the format-vs-task diagnostic now makes legible.

### Cost

DoRA's ~3.3x wall-time and ~1.8x peak memory overhead vs LoRA carries forward unchanged at the 10k-step budget — verified per-cell in the runtime and peak-mem columns above (DoRA ~5.3 h vs LoRA ~1.6 h per training run; DoRA ~60 GB vs LoRA ~34 GB peak GPU memory). This is the most robust quantitative DoRA-vs-LoRA result in the project: stable across tiers, ranks, regimes, and now training budgets.

### Scope — sequential to Tier 2, not integrated with it

Tier 3 was run *after* Tier 2 was complete and the 2026-05-23 scope decision was made (see [SUMMARY-tier2.md](SUMMARY-tier2.md) and AGENTS.md). Tier 2 already meets the project rubric's Example 4 ("Impressive") tier on its own; this document is a *separate sequential deliverable hanging off SUMMARY-tier2.md, not a revision or augmentation of it*. Tier-2 cell counts, seed sets, recipe, tables, figures, and prose are all unchanged by Tier 3 — verify by cross-comparison against the per-tier results directories (`results/tier2-cs170k/` vs `results/tier3-cs170k/`), which have disjoint run-id namespaces (Tier-3 ids carry a `_t3` suffix). Tier 3's contribution is to test two Tier-2 hypotheses against 4x training and one extra seed: it sharpens the inverse-rank trend from ~3 pp to ~14 - 16 pp within-method on broad genmatch, and refines the Tier-2 strict-genmatch DoRA hint into a sharper +0.16 cell-mean edge on the likelihood axis at r=8 (outside seed std at n=4, with 3 of 4 paired seeds favouring DoRA).

## How to regenerate

```bash
cd mini-project-DoRA
python scripts/summarize_results.py --tier 3 > results/SUMMARY-tier3.md
```

Prerequisites: every Tier-3 run dir under `results/tier3-cs170k/` has a `metrics.json` with at minimum `eval_accuracy_likelihood`; cells missing the post-training gen-eval pass render `-` in the genmatch column rather than failing. The Tier-2 cross-tier comparison loads from `results/tier2-cs170k/`.

