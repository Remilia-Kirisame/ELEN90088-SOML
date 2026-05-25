# mini-project-DoRA

A reproduction-with-extension of the **DoRA** paper (Liu et al., *ICML 2024*) for ELEN90088 — System Optimisation and Machine Learning (UniMelb, 2026). Project 2 (Hands-on LLMs), Part 6 (mini-project).

Framed as a **regime-contrast generalization study** rather than a strict replication: paper used LLaMA-base, we use Mistral-7B-Instruct (the open-access model already in the course toolchain); paper trained ~32k steps on `commonsense_170k`, we step-cap at 2,500 within the course compute budget. The headline question stays: *does DoRA outperform LoRA more at low rank, as the paper claims?*

- **Paper:** Liu et al. 2024, [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- **Upstream code:** [NVlabs/DoRA](https://github.com/NVlabs/DoRA) (commit `7e2f10a` pinned in [DoRA_REFERENCE.md](DoRA_REFERENCE.md))
- **Course brief:** [Project_2_Hands-on_LLMs.md](../Project-Description/Project_2_Hands-on_LLMs.md)

## TL;DR — headline findings

(Full numbers in [results/SUMMARY-tier2.md](results/SUMMARY-tier2.md). 36 training runs + 1 zero-shot baseline. Mean ± std over 3 seeds: `{42, 1, 2}`. A separate Tier-3 enrichment — 24 cs170k runs at 10k steps × 4 seeds `{114, 514, 1919, 810}` — is reported in [results/SUMMARY-tier3.md](results/SUMMARY-tier3.md); see also the [results index](results/SUMMARY.md).)

1. **DoRA does not meaningfully outperform LoRA on task accuracy** in this setup. The DoRA−LoRA gap is within seed std at every rank in BoolQ and at $r \in \{4, 8\}$ in cs170k. The only direction-consistent gap is at cs170k $r{=}16$ where DoRA *loses* −0.032 to LoRA (still below formal significance at $n{=}3$).
2. **DoRA shows a small directional edge on format-adaptation strength** — the *strict-parser* cs170k genmatch — of +0.056 at $r \in \{8, 16\}$. Seed std is large (0.16–0.35), so within noise at $n{=}3$, but the direction matches the paper's central hypothesis.
3. **DoRA cost overhead is firmly reconfirmed:** ~3.3× wall-time, ~1.8× peak GPU memory at every rank, both regimes.
4. **Methodological discovery:** the strict-parser cs170k genmatch implicitly conflated *task accuracy* and *format-adaptation strength*. Broadening the parser (`parse_yes_no_or_true_false`) disentangled the two. Both views are reported. See `results/SUMMARY-tier2.md` and `figures/format_adaptation.png` for the contrast.

Figures: `figures/rank_sensitivity.png`, `figures/format_adaptation.png`, `figures/loss_curves.png`.

### Tier 3 enrichment (10k cs170k steps × 4 seeds, 24 runs)

Tier 2 alone meets the project rubric. Tier 3 is an enrichment pass that pushes training closer to the paper's ~32k-step regime (we land at 10k = 31%) and adds an extra seed for tighter statistics. Three findings worth pulling out (full numbers + interpretation in [results/SUMMARY-tier3.md](results/SUMMARY-tier3.md), visuals under `figures/tier3/`):

1. **The Tier-2 inverse-rank trend sharpens.** At 4× training, the r=4-vs-r=16 broad-genmatch gap widens from Tier 2's ~3 pp into **~14 pp within LoRA** and **~16 pp within DoRA** (roughly a 5× amplification of Tier 2's signal). r=4 cells sit at or above the 0.82 zero-shot baseline; r=16 cells drop into 0.66–0.71. More PEFT capacity = more drift from BoolQ-optimal toward the broader cs170k mix.
2. **Methodological diagnostic, scaled up.** The BoolQ-yes/no likelihood probe *collapses* at every Tier-3 cell (Δ = −0.10 to −0.40), but broad-parser genmatch shows only modest task-accuracy loss (Δ = −0.006 to −0.12). The model isn't broken; it has fully switched to cs170k's true/false vocabulary. The format-vs-task split Tier 2 introduced as a footnote becomes load-bearing at Tier 3 — **genmatch is the right Tier-3 metric; likelihood at Tier 3 measures format preservation, not task accuracy**.
3. **DoRA-vs-LoRA, refined.** Tier 2's task-accuracy null replicates at n=4 — DoRA-LoRA genmatch gap within seed std at every rank. On the format-preservation axis (likelihood), the DoRA r=8 cell shows a +0.16 edge over LoRA r=8, well outside seed std and consistent across all 4 seeds: DoRA at mid-rank resists the cs170k format takeover where LoRA capitulates. Reframes Tier 2's "+0.056 strict-genmatch direction-consistent edge" claim with sharper statistical support and a cleaner metric story.

DoRA's ~3.3× wall-time / ~1.8× peak-memory cost overhead is reconfirmed at the 10k-step budget.

## What's in this repository

```
mini-project-DoRA/
├── pyproject.toml, uv.lock, .python-version    # uv project metadata (torch+cu124 pinned for Linux GPU)
├── DoRA_REFERENCE.md                            # paper link + upstream commit pin
├── src/dora_mini/                               # importable Python package
│   ├── paths.py            # env-var path helpers (PROJECT_DIR, UV_PROJECT_ENVIRONMENT, HF_HOME)
│   ├── configs.py          # single source of truth: Tier-2 grid (36 runs) + Tier-3 grid (24 runs, cs170k-only)
│   ├── data.py             # BoolQ + commonsense_170k loaders + chat-template formatting
│   ├── answer_parsing.py   # parse_yes_no, parse_true_false, parse_yes_no_or_true_false
│   ├── models.py           # load_tokenizer, load_base_model, wrap_peft (use_dora switch)
│   ├── train.py            # run_training(config_path) — full training pipeline
│   ├── eval.py             # evaluate_boolq (likelihood) + evaluate_boolq_generate (genmatch)
│   ├── summarize.py        # Tier-2 seed aggregation → SUMMARY-tier2.md tables
│   └── summarize_tier3.py  # Tier-3 aggregation + cross-tier comparison → SUMMARY-tier3.md
├── scripts/                                     # CLI entrypoints
│   ├── smoke_test.py       # env + model + PEFT sanity check (no checkpoint saved)
│   ├── gen_configs.py      # writes the 36 Tier-2 YAMLs into configs/tier2-*/
│   ├── train.py            # python scripts/train.py --config <yaml>
│   ├── evaluate.py         # generation-eval over a saved adapter (or --zero-shot for baseline)
│   ├── summarize_results.py# `--tier {2,3}` flag; aggregates results/**/metrics.json → results/SUMMARY-tier{2,3}.md
│   ├── sbatch_quick.sh     # short-queue SLURM template (gpu-a100-short, 30 min)
│   ├── sbatch_train.sh     # long-queue SLURM template (gpu-h100, 4 hr)
│   └── local/setup_env.example.sh  # committed env-var template (real setup_env.sh is gitignored)
├── configs/                                     # one YAML per experiment
│   ├── tier1/                                   # 6 historical Tier-1 configs (single-seed rank sweep)
│   ├── tier2-boolq/                             # 18 Tier-2 Phase-1 configs (BoolQ, 3 seeds)
│   ├── tier2-cs170k/                            # 18 Tier-2 Phase-2 configs (cs170k, 3 seeds)
│   └── tier3-cs170k/                            # 24 Tier-3 cs170k configs (10k steps, 4 seeds)
├── results/                                     # text artifacts only (adapters/checkpoints are gitignored)
│   ├── SUMMARY.md                               # index pointing to per-tier summaries
│   ├── SUMMARY-tier2.md                         # canonical Tier-2 deliverable: 12 cells + zero-shot + 4 gap tables + prose
│   ├── SUMMARY-tier3.md                         # Tier-3 enrichment: 6 cs170k cells at 10k steps + cross-tier deltas
│   ├── tier1/                                   # historical Tier-1 metrics.json + configs
│   ├── tier2-boolq/                             # Tier-2 Phase-1 metrics.json + train.log per run
│   ├── tier2-cs170k/                            # Tier-2 Phase-2 metrics.json + train.log per run
│   │   └── _strict_genmatch_pre_fix.json        # snapshot of pre-parser-fix strict genmatch
│   ├── tier2-baseline/                          # zero-shot baseline metrics.json
│   └── tier3-cs170k/                            # Tier-3 10k-step cs170k metrics.json + train.log per run (24 runs, _t3 suffix)
├── figures/                                     # three audience-ready figures
├── notebooks/                                   # jupytext-paired analysis (.py is source of truth)
└── tests/                                       # pytest — 61 tests, all passing on Mac (no GPU stack)
```

Adapters (`adapter/`, `checkpoint-*/`, `trainer_out/`, `*.safetensors`) and the `data/` cache are gitignored.

## Setup

This project has two modes:

- **Mac (or any laptop, no GPU)** — for running tests and the analysis notebooks. The 61 pytest tests cover paths, configs (Tier 2 + Tier 3), parsing, dataset utilities, and summarization (`summarize` + `summarize_tier3`) — they don't import torch.
- **Linux GPU host (Spartan HPC, or your own machine)** — for training, evaluation, and the smoke test. Requires a CUDA Linux GPU (`torch==2.6.0+cu124` is pinned in `pyproject.toml`; the cu124 wheel is Linux-x86_64 only).

### System requirements (for training)

The observed peaks on Spartan H100 (from `results/SUMMARY-tier2.md`):

- **GPU VRAM:** ~60 GB peak for DoRA training (any rank, both regimes); ~34 GB peak for LoRA. **Will not fit on consumer GPUs** (24 GB RTX 4090 / A10G); you need H100 80 GB, A100 80 GB, H200, or equivalent. LoRA-only experiments fit comfortably on 40 GB A100 / L40S.
- **CUDA driver:** 12.4-compatible (matches the pinned `torch==2.6.0+cu124` wheel). Older drivers (12.0–12.3) fail at `import torch` with a cryptic CUDA-version mismatch.
- **System RAM:** ≥ 32 GB. Headroom is mainly for the HF dataloader; the sbatch template requests 32 GB.
- **Disk:** ~15 GB for the Mistral-7B-Instruct weights (HF cache, bfloat16), ~150 MB for `commonsense_170k.json`, plus a few GB for adapters + logs. Route these to project storage, not `$HOME` — see [`docs/spartan-ood-setup.md`](docs/spartan-ood-setup.md).
- **Time (Tier 2):** the full 36-run sweep takes **~22 GPU-hours of training + ~3 GPU-hours of eval = ~25 GPU-hours on H100** (~3× on A100). Per-run: LoRA ~6 min BoolQ / ~27 min cs170k; DoRA ~20 min BoolQ / ~90 min cs170k.
- **Time (Tier 3):** the 24-run cs170k enrichment at 10k steps takes **~84 GPU-hours of training + ~30 GPU-hours of gen-eval ≈ ~114 GPU-hours on H100** (12 LoRA × ~98 min + 12 DoRA × ~318 min for training; gen-eval ~75 min × 24). Per-run: LoRA cs170k 10k ≈ 98 min; DoRA cs170k 10k ≈ 318 min. Observed wall-clock for our submission: 11h42m training + 2h40m gen-eval ≈ 14h22m total, with 8-10 jobs running in parallel on `gpu-h100`.

The Mac-side smoke test has no GPU requirement.

### Quick smoke test (Mac, ~10 seconds)

```bash
git clone <this-repo>
cd ELEN90088-SOML/mini-project-DoRA
pytest -q
```

You should see `61 passed`. This validates the path helpers, config grid (Tier-2 + Tier-3), parsers, dataset utilities, and summarization logic (`summarize` + `summarize_tier3`) without needing the LLM stack.

### Full setup (Linux GPU host)

This is the path for actually running training. The recipe assumes a SLURM scheduler (matches Spartan) but adapts to local GPU machines straightforwardly.

```bash
# 1. Clone
git clone <this-repo>
cd ELEN90088-SOML/mini-project-DoRA

# 2. Copy the env template and fill in your paths
cp scripts/local/setup_env.example.sh scripts/local/setup_env.sh
$EDITOR scripts/local/setup_env.sh   # set <your-project-id> for project storage

# 3. Source env + create uv venv (deps from pyproject.toml + uv.lock)
source scripts/local/setup_env.sh    # sets PROJECT_DIR, HF_HOME, UV_PROJECT_ENVIRONMENT
uv sync                              # creates venv at $UV_PROJECT_ENVIRONMENT, installs torch+cu124 + transformers + peft

# 4. Smoke test (full stack — confirms model loads + PEFT wraps)
python scripts/smoke_test.py
```

The `setup_env.example.sh` routes:

- code → `$HOME` (NFS-friendly, small quota OK for source)
- venv + HF cache → project storage (`$PUNIM_MINI`) — bulky model weights live here

On a non-SLURM machine you can skip the `sbatch_*.sh` scripts and just run `python scripts/train.py --config <yaml>` directly under your venv.

For **OOD / Code Server / Jupyter** users on a Spartan-style HPC, see [`docs/spartan-ood-setup.md`](docs/spartan-ood-setup.md) for the pre-run text (env vars set before the OOD launcher boots the node) and the rationale (Jupyter kernel env propagation + cache hygiene). Recommended for cache hygiene, required if you'll use Jupyter from inside Code Server.

## Reproduce the results

The full Tier-2 sweep is 36 training runs — **~22 GPU-hours of training + ~3 GPU-hours of evaluation on H100** (~3× on A100). Per-run breakdown is in the System requirements section above. Each run writes a `metrics.json` that the summarizer aggregates into `results/SUMMARY-tier2.md`.

### Step 1 — Generate the config grid

```bash
python scripts/gen_configs.py
```

This writes the 36 Tier-2 YAMLs into `configs/tier2-boolq/` and `configs/tier2-cs170k/`. The grid is the single source of truth in `src/dora_mini/configs.py:all_configs()`.

### Step 2 — Train

**Before training: download `commonsense_170k.json` (Phase 2 only).** The BoolQ split is fetched automatically by `datasets.load_dataset("boolq")`, but cs170k is a JSON file from the LLM-Adapters project that you must place at `data/commonsense_170k.json`:

```bash
mkdir -p data
curl -L -o data/commonsense_170k.json \
  https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/ft-training_set/commonsense_170k.json
```

(For tight reproducibility, pin to a specific LLM-Adapters commit instead of `main`; the file has been stable but `main` is not version-locked.) The file is ~150 MB. Skip this download if you only want to run BoolQ (Phase 1).

On Spartan / SLURM:

```bash
for cfg in configs/tier2-boolq/*.yaml configs/tier2-cs170k/*.yaml; do
    sbatch scripts/sbatch_train.sh "$cfg"
done
```

On a local GPU machine:

```bash
for cfg in configs/tier2-boolq/*.yaml configs/tier2-cs170k/*.yaml; do
    python scripts/train.py --config "$cfg"
done
```

Each run writes:

- `results/<tier>/<run_id>/metrics.json` — run config snapshot + `code_commit` + loss curves + likelihood accuracy + runtime/memory
- `results/<tier>/<run_id>/train.log` — clean per-run stdout
- `results/<tier>/<run_id>/adapter/` — the trained adapter weights (gitignored)

### Step 3 — Generation evaluation

The training pass records *likelihood* accuracy (first-token Yes/No logprob, parser-independent). To add the paper-style *generation exact-match* metric, run `evaluate.py` over each saved adapter:

```bash
for d in results/tier2-boolq/*/ results/tier2-cs170k/*/; do
    python scripts/evaluate.py --run "$d"
done
```

For the zero-shot baseline (base Mistral-Instruct, no adapter):

```bash
python scripts/evaluate.py --zero-shot --model mistralai/Mistral-7B-Instruct-v0.3
```

To inspect raw model outputs without writing metrics (useful for the parser-vs-real-collapse diagnostic that surfaced our two-views finding), add `--debug-print N` to either form above:

```bash
python scripts/evaluate.py --run results/tier2-cs170k/<run_id> --debug-print 10
python scripts/evaluate.py --zero-shot --model mistralai/Mistral-7B-Instruct-v0.3 --debug-print 10
```

### Step 4 — Aggregate into SUMMARY-tier2.md

```bash
python scripts/summarize_results.py --tier 2 > results/SUMMARY-tier2.md
```

This reads every Tier-2 `metrics.json` under `results/**/`, builds the 12 mean ± std cells (BoolQ + cs170k × LoRA/DoRA × 3 ranks), and renders the four gap tables (likelihood / broad genmatch per regime, plus the strict-genmatch gap for cs170k). The strict-genmatch values are loaded from `_strict_genmatch_pre_fix.json` (see caveats below).

For the Tier-3 enrichment (10k-step cs170k, 4 seeds, 24 runs), pass `--tier 3` instead:

```bash
python scripts/summarize_results.py --tier 3 > results/SUMMARY-tier3.md
```

The Tier-3 renderer also loads Tier-2 cs170k cells for cross-tier comparison tables.

### Step 5 — Regenerate figures (optional)

The three figures in `figures/` are produced by the paired jupytext notebook at `notebooks/analysis.{py,ipynb}`. After re-running the sweep (so `results/**/metrics.json` reflects your new data):

1. **Sync the `.ipynb` from the `.py` source-of-truth** (jupytext is in the `dev` extras):

   ```bash
   jupytext --sync notebooks/analysis.ipynb
   ```

2. **Execute the notebook end-to-end.** Two paths, pick whichever fits your environment:

   - *Interactive (recommended).* Open `notebooks/analysis.ipynb` in VSCode (with the Python + Jupyter extension) or JupyterLab and Run All. The notebook calls `fig.savefig(...)` for each figure, so the PNGs are written out as cells run. No extra install needed beyond the `dev` extras.
   - *Command-line, if you have `nbconvert` installed separately* (it's intentionally not in `dev` to keep the project install minimal):

     ```bash
     jupyter nbconvert --execute --inplace notebooks/analysis.ipynb
     ```

     `nbconvert` ships with full Jupyter installs (`pip install jupyter` or any distro's Jupyter package) — convenient if you want to regen figures non-interactively from a CI or batch context.

Either way, outputs land in `figures/rank_sensitivity.png`, `figures/format_adaptation.png`, `figures/loss_curves.png`. The committed PNGs are already in sync with the committed `metrics.json`, so this step is only needed if you've changed the data.

**For Tier-3 figures**, the parallel notebook is `notebooks/analysis_tier3.{py,ipynb}` and it writes to `figures/tier3/` (subdirectory — keeps the canonical Tier-2 PNGs at their committed paths). Same workflow:

```bash
jupytext --sync notebooks/analysis_tier3.ipynb
# then Run All in VSCode/Jupyter, or:
jupyter nbconvert --execute --inplace notebooks/analysis_tier3.ipynb
```

Outputs: `figures/tier3/rank_sensitivity.png`, `figures/tier3/format_drift.png`, `figures/tier3/loss_curves.png`.

## Reproduce the Tier-3 enrichment

The Tier-3 enrichment is 24 cs170k runs at 10k steps with seeds `{114, 514, 1919, 810}` — **~84 GPU-hours of training + ~30 GPU-hours of gen-eval ≈ ~114 GPU-hours on H100**. Same code path as Tier 2; the `--tier 3` flag and the `_t3` run-id suffix route everything to `configs/tier3-cs170k/` and `results/tier3-cs170k/` without touching Tier-2 artifacts. Tier 3 is *sequential* to Tier 2 (run after Tier 2 was complete), not integrated with it — see `results/SUMMARY-tier3.md` for the framing.

**Prerequisite — `commonsense_170k.json` must already be on disk at `data/commonsense_170k.json`.** Use the download command from the Tier-2 reproduce path (Step 2 above) if you haven't run Tier-2 training yet. Tier 3 reads from the same gitignored `data/` location.

### Step T3.1 — Generate the Tier-3 config grid

```bash
python scripts/gen_configs.py --tier 3
```

Writes 24 YAMLs into `configs/tier3-cs170k/`. Source of truth: `src/dora_mini/configs.py:all_tier3_configs()`.

### Step T3.2 — Train

Same `sbatch_train.sh` template (now bumped to `--time=08:00:00` to cover DoRA 10k runs that take ~5.3 h):

```bash
for cfg in configs/tier3-cs170k/*.yaml; do
    sbatch scripts/sbatch_train.sh "$cfg"
done
```

Or on a local GPU machine:

```bash
for cfg in configs/tier3-cs170k/*.yaml; do
    python scripts/train.py --config "$cfg"
done
```

Each run writes the same artifacts as Tier 2 (`metrics.json`, `train.log`, `adapter/`) into `results/tier3-cs170k/<run_id>_t3/`.

### Step T3.3 — Generation evaluation

Same pattern as Tier-2's Step 3, but over the `tier3-cs170k/` directories. **Requires the adapter weights from Step T3.2** — adapter directories under `results/tier3-cs170k/<run_id>_t3/adapter/` are gitignored (only `config.yaml`, `metrics.json`, `train.log` are committed), so a cold reproducer must run T3.2 before T3.3. The `--partition=gpu-h100 --time=02:00:00` CLI overrides matter (sbatch_quick.sh defaults to `gpu-a100-short` which caps at ~2 GPUs/user — queue-contention bottleneck — and 30 min, which a ~60-80 min gen-eval would TIMEOUT):

```bash
for d in results/tier3-cs170k/*/; do
    sbatch --partition=gpu-h100 --time=02:00:00 scripts/sbatch_quick.sh \
        python scripts/evaluate.py --run "${d%/}"
done
```

Appends `eval_accuracy_genmatch` to each existing `metrics.json`.

### Step T3.4 — Aggregate into SUMMARY-tier3.md

Already shown in Step 4 above:

```bash
python scripts/summarize_results.py --tier 3 > results/SUMMARY-tier3.md
```

The Tier-3 renderer also loads Tier-2 cs170k cells for the cross-tier (T3 − T2) comparison tables. **If your local `results/tier2-cs170k/` is empty** (e.g., a Tier-3-only reproducer who hasn't run Tier 2), the script still succeeds but the cross-tier delta tables will render empty rows. The Tier-3-only cells render correctly regardless.

### Step T3.5 — Regenerate the Tier-3 figures (optional)

See the "For Tier-3 figures" paragraph in Step 5 above — same workflow as Tier 2's notebook, but uses `notebooks/analysis_tier3.{py,ipynb}` and writes to `figures/tier3/`. Quick summary:

```bash
jupytext --sync notebooks/analysis_tier3.ipynb
# then Run All in VSCode/Jupyter, or:
jupyter nbconvert --execute --inplace notebooks/analysis_tier3.ipynb
```

Outputs: `figures/tier3/rank_sensitivity.png`, `figures/tier3/format_drift.png`, `figures/tier3/loss_curves.png`.

## Methodology brief

- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`. Not the paper's LLaMA-base. This is a deliberate generalization test: does the paper's claim survive on a different base?
- **PEFT library:** HuggingFace `peft` ≥0.13 with `LoraConfig(use_dora=True)` for DoRA. We do *not* use the upstream paper's bundled custom PEFT fork — see [DoRA_REFERENCE.md](DoRA_REFERENCE.md).
- **Sweep grid:** 2 methods (LoRA, DoRA) × 3 ranks (4, 8, 16) × 3 seeds (42, 1, 2) × 2 regimes (BoolQ, cs170k) = 36 runs. $\alpha/r = 2$ throughout (paper convention).
- **Two regimes:**
  - *Phase 1 — per-task BoolQ.* Full 9,427-example BoolQ train split. ~6 min per LoRA run, ~20 min per DoRA run on H100.
  - *Phase 2 — multi-task `commonsense_170k`.* Paper's 170k-example commonsense mix (BoolQ + 7 others). Step-capped at 2,500 — training loss converged by ~step 1,200, LR fully decayed by 2,500. Paper-faithful would be ~32k steps (~19 h per DoRA run), feasible on Spartan but excessive for a course-project budget.
- **Two evaluation metrics:**
  - *Likelihood* — first-token Yes/No logprob ratio. Parser-independent internal probe.
  - *Generation exact-match (`genmatch`)* — greedy decode + parser + exact match. Paper-style. Sensitive to format collapse.
- **Eval set:** Full BoolQ dev (3,270 examples) for both regimes, so the metric is comparable across them.
- **Compute:** UniMelb Spartan HPC, `gpu-h100` partition. Per-job `--time` in `sbatch_train.sh` is **8 h** (covers Tier-2 DoRA runs at ~90 min and Tier-3 DoRA runs at ~5.3 h with buffer). The partition itself allows up to 7 days; the 4 h cap we initially used was a self-imposed default in our sbatch template, not a partition limit. Tens of GPU-hours total across Tier 2 training + gen-eval + the parser-fix re-run; another ~114 GPU-hours for the Tier 3 enrichment.

The interpretation prose in `results/SUMMARY-tier2.md` splits findings into three views — *task accuracy*, *format adaptation*, and *cost* — and discusses the inverse-rank trend (low-rank cs170k training beats zero-shot on BoolQ; high-rank underperforms it). The Tier-3 summary (`results/SUMMARY-tier3.md`) amplifies the inverse-rank trend at 4× training and adds a method × rank interaction not visible at the Tier-2 budget.

## Caveats and known limitations

A few things worth knowing before you try to bit-reproduce numbers.

- **Strict-parser cs170k genmatch is a one-way snapshot.** The original strict-parser numbers were overwritten by the broad-parser re-run (when we discovered the parser was conflating task accuracy with format-adaptation). They are preserved verbatim in `results/tier2-cs170k/_strict_genmatch_pre_fix.json`, loaded by `summarize.load_strict_snapshot()`. Re-running `evaluate.py` today gives you broad-parser numbers, not strict.
- **Tier-1 configs predate the current code.** The 6 single-seed Tier-1 configs in `configs/tier1/` used `eval_size=500` (subset, for cost reasons) versus Tier-2's full 3,270-example BoolQ dev. The Tier-1 `metrics.json` files in `results/tier1/` are preserved as historical record. Re-running them today against current code will give comparable but not bit-identical numbers (random subsampling differs across runs).
- **$n{=}3$ seeds is low.** Especially for the strict-parser metric where seed std is 0.16–0.35, the +0.056 DoRA edge at $r \in \{8, 16\}$ is *direction-consistent* but below formal significance. More seeds would clarify. *Tier 3 partially addresses this with a separate n=4 enrichment at 10k steps; the refined picture is in [results/SUMMARY-tier3.md](results/SUMMARY-tier3.md).*
- **2,500 cs170k steps ≪ paper's ~32k.** The training loss converges by ~step 1,200 in our setup, but the model has less time to settle into the cs170k distribution than the paper allows. The DoRA strict-parser edge might cross significance under a paper-faithful budget. *Tier 3 pushes to 10k steps (31% of paper's 32k); the inverse-rank trend sharpens substantially. See [results/SUMMARY-tier3.md](results/SUMMARY-tier3.md).*
- **Different base model.** Mistral-7B-Instruct (instruction-tuned) vs the paper's LLaMA-base (raw pretrained). The starting point matters for PEFT studies — instruction-tuned models are already format-stable, which suppresses the kind of format-collapse failure mode the paper's "DoRA wins at low rank" finding is sensitive to.
- **Non-bit-deterministic.** Even with explicit seeds, cuDNN nondeterminism + dataloader ordering subtleties produce ~±0.005 variation across re-runs on the same config. The reported std captures this.

## Attribution and references

- **DoRA paper:** Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML 2024. [arXiv:2402.09353](https://arxiv.org/abs/2402.09353).
- **Upstream code:** [NVlabs/DoRA](https://github.com/NVlabs/DoRA), commit `7e2f10a` pinned for traceability. Their custom PEFT fork is *not* used at runtime; we use HuggingFace `peft`'s `use_dora=True` flag.
- **Datasets:** [`google/boolq`](https://huggingface.co/datasets/google/boolq) (Clark et al., 2019); [`commonsense_170k`](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/ft-training_set) via the LLM-Adapters mix.
- **Libraries:** `transformers`, `peft`, `accelerate`, `datasets` (HuggingFace); `bitsandbytes`; `torch`. Versions pinned in `pyproject.toml` + `uv.lock`.
- **Compute:** University of Melbourne Research Computing Services (Spartan HPC), gpu-h100 partition.
- **Course:** ELEN90088 System Optimisation and Machine Learning, UniMelb, Semester 1 2026.

## License and academic-integrity note

This is a coursework repository, shared as a personal record of how the study was done. See the root [LICENSE.md](../LICENSE.md) for usage. If you are currently enrolled in ELEN90088, please consult your subject's academic integrity policy before referring to any material here — copying submissions is a breach of UniMelb's Academic Integrity rules.

The upstream DoRA code under [NVlabs/DoRA](https://github.com/NVlabs/DoRA) is governed by its own license (NVIDIA Source Code License). Cite the paper rather than this reproduction for the DoRA method itself.
