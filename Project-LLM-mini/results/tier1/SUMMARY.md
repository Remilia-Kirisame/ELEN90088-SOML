# Tier 1 results summary

Generated 2026-05-16 (UTC) from `results/*_*/metrics.json` by `scripts/summarize_results.py`. 6 runs: LoRA vs DoRA at r ∈ {4, 8, 16} on BoolQ + Mistral-7B-Instruct-v0.3.

## Per-run table

| run_id | method | r | α | α/r | trainable | % of full | accuracy | NLL (yes/no) | runtime (s) | runtime (min) | peak mem (GB) | slurm job | started (UTC) | finished (UTC) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `dora_mistral7b_boolq_r4` | DORA | 4 | 8 | 2 | 11,862,016 | 0.163% | 0.8740 | 10.6741 | 1183 | 19.7 | 59.4 | `25035695` | 2026-05-16 13:21 UTC | 2026-05-16 13:41 UTC |
| `dora_mistral7b_boolq_r8` | DORA | 8 | 16 | 2 | 22,347,776 | 0.307% | 0.8880 | 11.0463 | 1167 | 19.4 | 59.6 | `25031133` | 2026-05-16 11:10 UTC | 2026-05-16 11:30 UTC |
| `dora_mistral7b_boolq_r16` | DORA | 16 | 32 | 2 | 43,319,296 | 0.594% | 0.8800 | 11.4465 | 1164 | 19.4 | 59.9 | `25035696` | 2026-05-16 13:27 UTC | 2026-05-16 13:46 UTC |
| `lora_mistral7b_boolq_r4` | LORA | 4 | 8 | 2 | 10,485,760 | 0.144% | 0.8800 | 10.2051 | 364 | 6.1 | 33.3 | `25035693` | 2026-05-16 13:11 UTC | 2026-05-16 13:17 UTC |
| `lora_mistral7b_boolq_r8` | LORA | 8 | 16 | 2 | 20,971,520 | 0.289% | 0.8820 | 10.7806 | 353 | 5.9 | 33.5 | `25028853` | 2026-05-16 10:04 UTC | 2026-05-16 10:09 UTC |
| `lora_mistral7b_boolq_r16` | LORA | 16 | 32 | 2 | 41,943,040 | 0.575% | 0.8840 | 11.6086 | 357 | 6.0 | 33.9 | `25035694` | 2026-05-16 13:20 UTC | 2026-05-16 13:25 UTC |

## DoRA − LoRA accuracy gap by rank

| r | LoRA accuracy | DoRA accuracy | Δ (DoRA − LoRA) |
|---:|---:|---:|---:|
| 4 | 0.8800 | 0.8740 | -0.0060 |
| 8 | 0.8820 | 0.8880 | +0.0060 |
| 16 | 0.8840 | 0.8800 | -0.0040 |

## Cost ratios (DoRA vs LoRA at the same r)

| r | runtime ratio | peak mem ratio | param overhead (DoRA − LoRA) |
|---:|---:|---:|---:|
| 4 | 3.25× | 1.78× | +1,376,256 |
| 8 | 3.31× | 1.78× | +1,376,256 |
| 16 | 3.26× | 1.77× | +1,376,256 |

## Per-step training loss

| run_id | step 100 | step 200 | step 300 | step 400 | step 500 |
|---|---:|---:|---:|---:|---:|
| `dora_mistral7b_boolq_r4` | 1.4131 | 0.1148 | 0.0592 | 0.0247 | 0.0036 |
| `dora_mistral7b_boolq_r8` | 1.1713 | 0.1151 | 0.0381 | 0.0169 | 0.0041 |
| `dora_mistral7b_boolq_r16` | 0.9808 | 0.1135 | 0.0249 | 0.0115 | 0.0019 |
| `lora_mistral7b_boolq_r4` | 1.5528 | 0.1212 | 0.0614 | 0.0285 | 0.0039 |
| `lora_mistral7b_boolq_r8` | 1.3084 | 0.1096 | 0.0441 | 0.0138 | 0.0028 |
| `lora_mistral7b_boolq_r16` | 1.0652 | 0.1047 | 0.0313 | 0.0042 | 0.0013 |

## Shared configuration

- **Model:** `mistralai/Mistral-7B-Instruct-v0.3` (bfloat16)
- **Dataset:** `boolq` (train_size=2000, eval_size=500, max_length=512)
- **PEFT:** `target_modules=all-linear`, `dropout=0.05` (r and α vary per run)
- **Training:** batch_size=4 × grad_accum=4 (effective batch = 16), lr=5e-05, num_steps=500, warmup_steps=50, seed=42

## Environment

- Python 3.10.20 · torch 2.6.0+cu124 · transformers 4.57.6 · peft 0.19.1
- GPU: NVIDIA H100 80GB HBM3 (Spartan `gpu-h100` partition)

## Statistical caveat

Single seed per config, `eval_size=500`. Binomial standard error on a single accuracy measurement at $p \approx 0.88$:

$$
\mathrm{SE} = \sqrt{\frac{p(1-p)}{n}} = \sqrt{\frac{0.88 \cdot 0.12}{500}} \approx 0.0145
$$

So per-measurement SE ≈ ±1.45 pp at 1σ, ±2.9 pp at 2σ. Every observed LoRA-vs-DoRA gap in our table (≤ 0.6 pp) is well inside one SE — cannot be statistically distinguished from zero without multiple seeds.

## How to regenerate

```bash
cd Project-LLM-mini && python scripts/summarize_results.py > results/SUMMARY.md
```

Re-run after adding new entries under `results/<run_id>/metrics.json`; the script picks up new runs by glob.

