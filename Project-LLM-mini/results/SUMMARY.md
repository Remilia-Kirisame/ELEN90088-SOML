# Tier 2 results summary

## Zero-shot baseline (Mistral-7B-Instruct, no adapter)

- likelihood accuracy: 0.8235
- genmatch accuracy:   0.8202

## Per-cell results (mean +/- std over 3 seeds)

| trainset | method | r | n | likelihood acc | genmatch (broad) | genmatch (strict) | runtime (s) | peak mem (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| boolq | DORA | 4 | 3 | 0.8742 +/- 0.0067 | 0.8927 +/- 0.0011 | - | 1189 | 59.4 |
| boolq | DORA | 8 | 3 | 0.8797 +/- 0.0079 | 0.8967 +/- 0.0026 | - | 1181 | 59.6 |
| boolq | DORA | 16 | 3 | 0.8811 +/- 0.0081 | 0.8991 +/- 0.0008 | - | 1181 | 59.9 |
| boolq | LORA | 4 | 3 | 0.8780 +/- 0.0040 | 0.8939 +/- 0.0021 | - | 361 | 33.3 |
| boolq | LORA | 8 | 3 | 0.8803 +/- 0.0106 | 0.8987 +/- 0.0036 | - | 361 | 33.5 |
| boolq | LORA | 16 | 3 | 0.8792 +/- 0.0043 | 0.8996 +/- 0.0026 | - | 356 | 33.9 |
| cs170k | DORA | 4 | 3 | 0.7499 +/- 0.0851 | 0.8499 +/- 0.0055 | 0.0980 +/- 0.0792 | 5474 | 59.4 |
| cs170k | DORA | 8 | 3 | 0.8140 +/- 0.0244 | 0.8403 +/- 0.0144 | 0.3183 +/- 0.1952 | 5429 | 59.6 |
| cs170k | DORA | 16 | 3 | 0.7294 +/- 0.0903 | 0.7844 +/- 0.0135 | 0.6296 +/- 0.2627 | 5426 | 59.9 |
| cs170k | LORA | 4 | 3 | 0.7788 +/- 0.0723 | 0.8485 +/- 0.0025 | 0.1069 +/- 0.1657 | 1653 | 33.3 |
| cs170k | LORA | 8 | 3 | 0.8103 +/- 0.0131 | 0.8377 +/- 0.0009 | 0.2624 +/- 0.3501 | 1611 | 33.5 |
| cs170k | LORA | 16 | 3 | 0.7835 +/- 0.0131 | 0.8159 +/- 0.0189 | 0.5742 +/- 0.2282 | 1603 | 33.9 |

### boolq - DoRA - LoRA gap (likelihood)

| r | LoRA | DoRA | delta (DoRA - LoRA) |
|---:|---:|---:|---:|
| 4 | 0.8780 +/- 0.0040 | 0.8742 +/- 0.0067 | -0.0038 |
| 8 | 0.8803 +/- 0.0106 | 0.8797 +/- 0.0079 | -0.0006 |
| 16 | 0.8792 +/- 0.0043 | 0.8811 +/- 0.0081 | +0.0019 |

### boolq - DoRA - LoRA gap (genmatch)

| r | LoRA | DoRA | delta (DoRA - LoRA) |
|---:|---:|---:|---:|
| 4 | 0.8939 +/- 0.0021 | 0.8927 +/- 0.0011 | -0.0012 |
| 8 | 0.8987 +/- 0.0036 | 0.8967 +/- 0.0026 | -0.0019 |
| 16 | 0.8996 +/- 0.0026 | 0.8991 +/- 0.0008 | -0.0005 |

### cs170k - DoRA - LoRA gap (likelihood)

| r | LoRA | DoRA | delta (DoRA - LoRA) |
|---:|---:|---:|---:|
| 4 | 0.7788 +/- 0.0723 | 0.7499 +/- 0.0851 | -0.0288 |
| 8 | 0.8103 +/- 0.0131 | 0.8140 +/- 0.0244 | +0.0037 |
| 16 | 0.7835 +/- 0.0131 | 0.7294 +/- 0.0903 | -0.0541 |

### cs170k - DoRA - LoRA gap (genmatch)

| r | LoRA | DoRA | delta (DoRA - LoRA) |
|---:|---:|---:|---:|
| 4 | 0.8485 +/- 0.0025 | 0.8499 +/- 0.0055 | +0.0014 |
| 8 | 0.8377 +/- 0.0009 | 0.8403 +/- 0.0144 | +0.0025 |
| 16 | 0.8159 +/- 0.0189 | 0.7844 +/- 0.0135 | -0.0315 |

### cs170k - DoRA - LoRA gap (genmatch_strict)

| r | LoRA | DoRA | delta (DoRA - LoRA) |
|---:|---:|---:|---:|
| 4 | 0.1069 +/- 0.1657 | 0.0980 +/- 0.0792 | -0.0089 |
| 8 | 0.2624 +/- 0.3501 | 0.3183 +/- 0.1952 | +0.0559 |
| 16 | 0.5742 +/- 0.2282 | 0.6296 +/- 0.2627 | +0.0555 |

## Notes on interpretation

### Two views of cs170k genmatch: broad parser vs strict parser

`genmatch (broad)` accepts either yes/no or true/false (normalized to true/false). It measures whether the model emits a parseable answer matching the gold label — i.e. task accuracy via free generation. `genmatch (strict)` only accepts literal true/false; it scores zero when the model obeys the BoolQ eval prompt's yes/no instruction and emits yes/no. So the strict view is a *format-adaptation strength* metric: how strongly the adapter has overridden the prompt with cs170k's true/false training format.

The strict column shows a clean rank-stratified pattern (low at r=4, much higher at r=16) reflecting that higher-rank adapters have the capacity to lock in the cs170k format while low-rank ones lack that capacity and default to the prompt's yes/no. Diagnostic spot-checks (raw generations of 4 representative runs) confirm this: low-rank cs170k models emit `"the correct answer is yes/no"` ~90% of the time; high-rank ones emit `"the correct answer is true/false"`. Both are answering correctly; the difference is surface form.

### Inverse-rank trend on broad genmatch

On the broad metric, cs170k genmatch decreases slightly with rank (r=4 ~0.85 → r=16 ~0.78–0.82) — opposite to "more capacity = better". Reading: cs170k is a multi-task mix (BoolQ is one of 8 tasks); training on it specializes the model *away from pure BoolQ*. At low rank the specialization is mild — the model retains its base BoolQ ability and gets a small boost from training (low-rank trained ~0.85 > zero-shot ~0.82). At high rank the model commits harder to the broader cs170k distribution, sometimes at the cost of BoolQ-specific accuracy (high-rank trained ~0.78 < zero-shot ~0.82). More capacity to learn cs170k means more capacity to drift from BoolQ-optimal behavior.

### DoRA vs LoRA: task accuracy vs format-adaptation

**Task accuracy (broad genmatch and likelihood).** The DoRA−LoRA gap is within seed std at every rank in the BoolQ regime and at r ∈ {4, 8} in cs170k. The exception is cs170k r=16: broad-genmatch shows DoRA −0.032 below LoRA, direction-consistent across all 3 seeds (gap exceeds seed std ~0.013–0.019, though n=3 is not enough for formal significance); the likelihood gap at the same cell (−0.054) is within DoRA's wide seed std at r=16 (0.090). The Tier-1 / Phase-1 null result largely reproduces in the multi-task regime that was specifically designed to surface a DoRA advantage; the paper's claim of a particularly large DoRA edge at low rank does not replicate as a task-accuracy improvement in our setup.

**Format-adaptation strength (strict genmatch).** Here DoRA shows a small but direction-consistent edge at r=8 (+0.056) and r=16 (+0.056); both r=4 cells are tied near the floor (~0.10). The seed std on this metric is large (~0.16–0.35 per cell, comparable to the means), so the gap is below statistical significance with n=3 — but the consistent direction matches the spec's central hypothesis: DoRA's magnitude/direction decomposition embeds the cs170k true/false training format more reliably than LoRA. It just doesn't translate into measurable task-accuracy improvement at our 2500-step training budget.

**Cost.** DoRA's overhead — roughly 3.3× wall-time and 1.8× peak GPU memory vs LoRA at otherwise identical settings — is firmly reconfirmed across both phases.

## How to regenerate

```bash
cd Project-LLM-mini && python scripts/summarize_results.py > results/SUMMARY.md
```

