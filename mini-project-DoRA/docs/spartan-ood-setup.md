# Spartan OOD / Code Server setup for `mini-project-DoRA`

Step-by-step environment setup for running this project on a Spartan-style HPC via **Open OnDemand (OOD)** with a **Code Server** session (the web-based VS Code) or Jupyter. Complements `scripts/local/setup_env.example.sh` — see the table below for how the two layers work together.

> If you are reproducing this from a fresh clone on a non-Spartan HPC, the same ideas apply: an OOD-style web launcher with a "Commands to run before launch" field, plus a project allocation on parallel storage. Adjust paths to your site.

## When to use this

Paste the [pre-run text](#pre-run-text-paste-into-spartan-ood) into the **"Commands to run before Code Server"** (or equivalent) field on the OOD launcher when requesting a *new* compute node. Then run the [post-launch verification block](#post-launch-run-inside-the-code-server-terminal) inside the Code Server terminal once the node boots.

The pre-run text is also the canonical reference if you want to rebuild your local OOD template from scratch — it documents every env var the project consumes, plus the supplementary cache-hygiene vars.

## Pre-run text vs `setup_env.sh` — complementary, not equivalent

A common misconception: the pre-run text and `source scripts/local/setup_env.sh` do "the same thing" so you only need one. They overlap on a critical subset, but each does something the other doesn't:

| | Pre-run text (this doc) | `scripts/local/setup_env.sh` |
|---|---|---|
| Sets `HF_HOME` / `HF_DATASETS_CACHE` / `HF_HUB_CACHE` | ✅ | ✅ (intentional overlap — double-insurance) |
| Sets `TORCH_HOME`, `TRITON_CACHE_DIR`, `MPLCONFIGDIR`, `TMPDIR`, `XDG_CACHE_HOME`, `IPYTHONDIR`, `PIP_CACHE_DIR` | ✅ | ❌ |
| Prepends `~/.local/bin` to `PATH` (so `uv` resolves in non-login shells) | ✅ | ❌ |
| Sets `PROJECT_DIR` | ❌ | ✅ |
| **Activates the venv** (`source $UV_PROJECT_ENVIRONMENT/bin/activate`) | ❌ | ✅ |

In practice:

- **`setup_env.sh` is mandatory.** Without it the venv is not activated, `python` resolves to system Python, and the code cannot find its paths.
- **Pre-run text is mandatory if you use Jupyter inside Code Server.** Jupyter kernels are forked from the OOD pre-run shell; env vars propagate at fork time only. Sourcing `setup_env.sh` later in a terminal pane has no effect on an already-running kernel — you'd silently miss `UV_PROJECT_ENVIRONMENT` (so imports may pick wrong wheels) and the cache routing inside the notebook.
- **Pre-run text is recommended otherwise** for clean cache hygiene. Without it, torch JIT compilation caches, triton kernels, matplotlib config, ipython history, etc. inherit whatever was in the launching shell — usually defaulting to `$HOME`, which has a 50 GB quota on Spartan. Harmless for small workloads, but bleeds quota and mixes with other projects.

For the **`sbatch` workflow** (the way training runs are launched), either layer alone usually leaks enough env vars into the submitting shell that `sbatch`'s inheritance picks up the right `UV_PROJECT_ENVIRONMENT` + `HF_HOME`. For Jupyter or for clean caches, you want both.

## Pre-run text (paste into Spartan OOD)

Replace `<your-project-id>` with your shared HPC allocation ID before pasting.

```bash
# === Project storage root ============================================
# All bulky data — venv, model weights, HF caches, torch JIT caches — lives under here, NOT $HOME (50 GB quota). The variable name is project-scoped ("PUNIM" is Spartan's project-allocation prefix; rename if your site uses a different convention).
export PUNIM_MINI=/data/gpfs/projects/<your-project-id>/dora-mini;

# === uv-managed venv =================================================
# Routes uv's per-project venv into project storage.
export UV_PROJECT_ENVIRONMENT="$PUNIM_MINI/venv";

# === HuggingFace cache routing =======================================
# HF_HOME is the root the transformers/datasets/peft libraries read at import time. The sub-dirs are set explicitly as a safety net in case some library reads them directly (e.g. older datasets versions).
export HF_HOME="$PUNIM_MINI/hf-cache";
export HF_DATASETS_CACHE="$HF_HOME/datasets";
export HF_HUB_CACHE="$HF_HOME/hub";
export HF_ASSETS_CACHE="$HF_HOME/assets";
export HF_MODULES_CACHE="$HF_HOME/modules";

# === Other library caches ============================================
# pip's wheel/index cache (uv uses its own cache; this catches anything that falls back to pip).
export PIP_CACHE_DIR="$PUNIM_MINI/pip-cache";
# torch C++/CUDA-extension build cache (compiled at first import; rebuilding into $HOME wastes quota).
export TORCH_HOME="$PUNIM_MINI/torch";
export TORCH_EXTENSIONS_DIR="$PUNIM_MINI/torch/extensions";
# Triton's kernel JIT compilation cache.
export TRITON_CACHE_DIR="$PUNIM_MINI/triton";
# Generic XDG fallback (some libs honour this; some don't).
export XDG_CACHE_HOME="$PUNIM_MINI/xdg/cache";
# IPython history + per-kernel state (matters if you use Jupyter from Code Server).
export IPYTHONDIR="$PUNIM_MINI/ipython";
# matplotlib config + font cache (otherwise ~/.matplotlib gets recreated on every run).
export MPLCONFIGDIR="$PUNIM_MINI/matplotlib";
# Process-scoped temp dir for the session (HF and torch sometimes spool large temp files).
export TMPDIR="$PUNIM_MINI/tmp";

# === PATH for uv =====================================================
# uv lives under ~/.local/bin. Non-interactive OOD shells don't source ~/.bashrc, so we put it on PATH explicitly. Needed for `uv sync` or `uv run` in this session.
export PATH="$HOME/.local/bin:$PATH";

# === Create target dirs ==============================================
mkdir -p "$PUNIM_MINI";
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$HF_MODULES_CACHE";
mkdir -p "$PIP_CACHE_DIR" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$IPYTHONDIR" "$MPLCONFIGDIR" "$TMPDIR";
```

### Notes on specific env vars

- **`TRANSFORMERS_CACHE` is intentionally not exported.** It was deprecated in `transformers >= 4.50` (raises a `FutureWarning`); the canonical layout is `HF_HOME`-based subdirs. Setting it would spam logs without doing anything new.
- **`SLURM_JOB_ID` is read by the project** (in [`src/dora_mini/train.py`](../src/dora_mini/train.py:92)) for run identification in `metrics.json`. It's **auto-set by SLURM** on `sbatch` runs (so don't export it manually). When running interactively, the code falls back to the literal `"interactive"`.
- **`PUNIM_MINI` is convention, not a code requirement.** It's just an intermediate that the lines below interpolate. If you prefer a different name (`PROJ_STORAGE`, `$PROJECT_ROOT`, etc.), rename it consistently in this block — the actual code paths (`paths.py`) only read `PROJECT_DIR`, `UV_PROJECT_ENVIRONMENT`, and `HF_HOME`.

## Post-launch (run inside the Code Server terminal)

Once the OOD node boots and Code Server opens:

```bash
cd ~/ELEN90088-SOML/mini-project-DoRA
source scripts/local/setup_env.sh
which python
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Expected output:

- `which python` → `/data/gpfs/projects/<your-project-id>/dora-mini/venv/bin/python`
- `python -c 'import torch; ...'` → `2.6.0+cu124 12.4 True` (on a GPU node)

Once those check out, you can:

- `python scripts/smoke_test.py` — env + model + PEFT sanity check (~2 min on H100)
- `sbatch scripts/sbatch_train.sh configs/<config>.yaml` — submit a training job
- `python scripts/evaluate.py --run results/<tier>/<run_id>` — generation-eval pass on a saved adapter (the script reads `config.yaml` from the run dir and locates `adapter/` itself)

## Env-var reference table

This is the single source of truth for what the project consumes and what's there for hygiene.

| Env var | Set by | Used by | Required? |
|---|---|---|---|
| `PROJECT_DIR` | `setup_env.sh` | `src/dora_mini/paths.py:project_dir()` | **Yes** — paths.py raises if unset |
| `UV_PROJECT_ENVIRONMENT` | `setup_env.sh` + pre-run | `paths.py:venv_dir()` + uv | **Yes** |
| `HF_HOME` | `setup_env.sh` + pre-run | `paths.py:hf_home()` + HF libraries | **Yes** |
| `HF_DATASETS_CACHE` | `setup_env.sh` + pre-run | `datasets` library | Recommended |
| `HF_HUB_CACHE` | `setup_env.sh` + pre-run | `huggingface_hub` | Recommended |
| `HF_ASSETS_CACHE` | pre-run | HF generic asset downloads | Optional (cache hygiene) |
| `HF_MODULES_CACHE` | pre-run | HF dynamic-module imports | Optional (cache hygiene) |
| `PIP_CACHE_DIR` | pre-run | `pip` wheel cache | Optional |
| `TORCH_HOME` | pre-run | torch model zoo / extensions | Optional |
| `TORCH_EXTENSIONS_DIR` | pre-run | torch C++/CUDA extension builds | Optional |
| `TRITON_CACHE_DIR` | pre-run | Triton JIT compilation cache | Optional |
| `XDG_CACHE_HOME` | pre-run | Generic XDG-compliant libraries | Optional |
| `IPYTHONDIR` | pre-run | IPython history + kernel state | Optional (matters for Jupyter) |
| `MPLCONFIGDIR` | pre-run | matplotlib config + font cache | Optional |
| `TMPDIR` | pre-run | Process temp files (HF, torch spill) | Optional |
| `PUNIM_MINI` | pre-run + `setup_env.sh` | Intermediate for all `*_CACHE_*` paths above | Convention only |
| `SLURM_JOB_ID` | SLURM (`sbatch`) | `src/dora_mini/train.py` — run logging in `metrics.json` | Auto-set; do not manually export |

## See also

- [`scripts/local/setup_env.example.sh`](../scripts/local/setup_env.example.sh) — committed template for the gitignored `setup_env.sh` (mandatory env vars + venv activation).
- [`README.md`](../README.md) — project overview and reproduce instructions.
- [`AGENTS.md`](../AGENTS.md) — repo conventions and Spartan workflow notes (gitignored / local-only).
