# Tier 2 results summary

## Per-cell results (mean +/- std over 3 seeds)

| trainset | method | r | n | likelihood acc | genmatch acc | runtime (s) | peak mem (GB) |
|---|---|---:|---:|---:|---:|---:|---:|
| boolq | DORA | 4 | 3 | 0.8742 +/- 0.0067 | 0.8927 +/- 0.0011 | 1189 | 59.4 |
| boolq | DORA | 8 | 3 | 0.8797 +/- 0.0079 | 0.8967 +/- 0.0026 | 1181 | 59.6 |
| boolq | DORA | 16 | 3 | 0.8811 +/- 0.0081 | 0.8991 +/- 0.0008 | 1181 | 59.9 |
| boolq | LORA | 4 | 3 | 0.8780 +/- 0.0040 | 0.8939 +/- 0.0021 | 361 | 33.3 |
| boolq | LORA | 8 | 3 | 0.8803 +/- 0.0106 | 0.8987 +/- 0.0036 | 361 | 33.5 |
| boolq | LORA | 16 | 3 | 0.8792 +/- 0.0043 | 0.8996 +/- 0.0026 | 356 | 33.9 |

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

## How to regenerate

```bash
cd Project-LLM-mini && python scripts/summarize_results.py > results/SUMMARY.md
```

