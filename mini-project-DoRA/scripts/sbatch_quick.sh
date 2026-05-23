#!/bin/bash
# Short-queue SLURM template for smoke tests + quick re-evals.
# Use: sbatch scripts/sbatch_quick.sh python scripts/smoke_test.py
# `exec "$@"` runs whatever positional args you pass — keeps the template generic.

#SBATCH --job-name=dora-quick
#SBATCH --output=results/_slurm/slurm-%j.out
#SBATCH --error=results/_slurm/slurm-%j.err
#SBATCH --time=00:30:00                          # short queue cap; smoke ~5 min, eval ~1 min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                                 # system RAM (not GPU VRAM); plenty for one 7B model
#SBATCH --partition=gpu-a100-short                # short queue gets faster start-of-job
#SBATCH --gres=gpu:1
set -euo pipefail

# Fail fast if env wasn't sourced — saves a GPU-hour of misconfigured run.
: "${UV_PROJECT_ENVIRONMENT:?source scripts/local/setup_env.sh before submitting}"
: "${HF_HOME:?HF_HOME must be set}"

mkdir -p "$HF_HOME"/{transformers,datasets,hub} results/_slurm
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# Run whatever the user passed as positional args.
exec "$@"
