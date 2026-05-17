#!/bin/bash
# Template for the gitignored setup_env.sh that lives next to this file.
# Copy this file to setup_env.sh and fill in your real values; setup_env.sh is gitignored — never commit your actual site paths.
#
# Why split storage: code lives in HOME (small quota, NFS, fine for editing) but model weights + venv are bulky and benefit from project storage (large quota, parallel filesystem). The env vars below route HF cache + uv venv to project storage.
#
# Usage on Spartan:
#   source scripts/local/setup_env.sh
#
# Placeholders below: replace <your-project-id> with your shared HPC allocation id.

# Code lives in $HOME (repo clone)
export PROJECT_DIR="$HOME/ELEN90088-SOML/Project-LLM-mini"

# Bulky data lives in project storage (HOME is quota-limited)
export PUNIM_MINI="/data/gpfs/projects/<your-project-id>/dora-mini"

# uv-managed venv lives in project storage too
export UV_PROJECT_ENVIRONMENT="$PUNIM_MINI/venv"

# HuggingFace cache routing
export HF_HOME="$PUNIM_MINI/hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"

# Ensure target dirs exist
mkdir -p "$PUNIM_MINI" "$HF_HOME"/{transformers,datasets,hub}

# Activate venv if it exists (skip on first-time setup before `uv sync`)
if [ -d "$UV_PROJECT_ENVIRONMENT" ]; then
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"
fi

# Sanity print
echo "[setup_env] PROJECT_DIR = $PROJECT_DIR"
echo "[setup_env] PUNIM_MINI  = $PUNIM_MINI"
echo "[setup_env] venv active: ${VIRTUAL_ENV:-(none yet)}"
