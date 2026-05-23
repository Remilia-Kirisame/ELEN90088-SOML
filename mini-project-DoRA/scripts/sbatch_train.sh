#!/bin/bash
# Long-queue SLURM template for training runs.
# Use: sbatch scripts/sbatch_train.sh configs/<config>.yaml
# Tier 1 single-config training takes ~6 min on H100 (~20-25 min on A100).

#SBATCH --job-name=dora-train
#SBATCH --output=results/_slurm/slurm-%j.out
#SBATCH --error=results/_slurm/slurm-%j.err
#SBATCH --time=04:00:00                          # generous cap; partition limit is 4h on gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                        # used by HF dataloader workers
#SBATCH --mem=32G                                 # system RAM, not GPU VRAM
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
set -euo pipefail

# Fail fast if env wasn't sourced — saves an H100-hour of misconfigured run.
: "${UV_PROJECT_ENVIRONMENT:?source scripts/local/setup_env.sh before submitting}"
: "${HF_HOME:?HF_HOME must be set}"

mkdir -p "$HF_HOME"/{transformers,datasets,hub} results/_slurm
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

CONFIG="$1"
echo "[$(date)] starting job $SLURM_JOB_ID on $(hostname) with $CONFIG"
python scripts/train.py --config "$CONFIG"
echo "[$(date)] done"
