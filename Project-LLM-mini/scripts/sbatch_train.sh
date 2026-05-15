#!/bin/bash
#SBATCH --job-name=dora-train
#SBATCH --output=results/_slurm/slurm-%j.out
#SBATCH --error=results/_slurm/slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
set -euo pipefail

: "${UV_PROJECT_ENVIRONMENT:?source scripts/local/setup_env.sh before submitting}"
: "${HF_HOME:?HF_HOME must be set}"

mkdir -p "$HF_HOME"/{transformers,datasets,hub} results/_slurm
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

CONFIG="$1"
echo "[$(date)] starting job $SLURM_JOB_ID on $(hostname) with $CONFIG"
python scripts/train.py --config "$CONFIG"
echo "[$(date)] done"
