# Results — index

Two summary documents live here, one per training-budget tier. Each is self-contained — open whichever matches the question you're asking.

- **[SUMMARY-tier2.md](SUMMARY-tier2.md)** — the **canonical project deliverable**. 36 training runs at the course-feasible 2500-step cs170k / 500-step BoolQ budget, 3 seeds `{42, 1, 2}`, 12 cells + zero-shot + four DoRA−LoRA gap tables + interpretation prose (three views: task accuracy, format adaptation, cost). This is the report-grade tier and meets the project rubric's Example 4 ("Impressive").

- **[SUMMARY-tier3.md](SUMMARY-tier3.md)** — **enrichment** at 10k cs170k steps × 4 seeds `{114, 514, 1919, 810}`, cs170k regime only (24 runs). Tests the two unresolved Tier-2 open questions: (a) does the +0.056 DoRA strict-genmatch direction-consistent edge clear noise at n=4 + 4× training, and (b) what happens as we step from a 7.8%-of-paper training budget toward 31% (still well short of the paper's full ~32k cs170k steps)? Headline: longer training drives both methods into a format-adaptation regime where the BoolQ-yes/no likelihood probe collapses, exposing a rank-dependent drift pattern that's invisible at Tier-2's shorter budget.

Tier 2 cells are unchanged by Tier 3 (separate `results/tier2-cs170k/` and `results/tier3-cs170k/` directories, separate seed sets, separate run-id namespaces via the `_t3` suffix), so both deliverables stand on their own and can be re-aggregated independently.

## How to regenerate

```bash
cd mini-project-DoRA
python scripts/summarize_results.py --tier 2 > results/SUMMARY-tier2.md
python scripts/summarize_results.py --tier 3 > results/SUMMARY-tier3.md
```

(See each summary's own "How to regenerate" footer for prerequisites — generally: training metrics.json + gen-eval pass complete for every cell.)
