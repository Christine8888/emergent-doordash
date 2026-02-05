#!/bin/bash
# Setup script for SLURM jobs - suzeva's version
# Uses Christine's shared venv (on juice - accessible from all nodes)

# Use Christine's venv on shared storage (works on sphinx and miso)
export VENV_DIR="/juice5b/scr5b/cye/.venv"
export UV_CACHE_DIR="/juice5b/scr5b/cye/.cache/uv"

# Use Christine's HF cache (has models pre-downloaded, faster storage)
export HF_HOME="/sphinx/u/cye/.cache/huggingface" # TODO should probably use /sphinx/u/suzeva/models
export HOME="/scr/suzeva"
mkdir -p "$HOME"
mkdir -p "$HF_HOME"

# Activate the shared venv
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Venv not found at $VENV_DIR"
    exit 1
fi

# Add project src/ to PYTHONPATH so submitit workers can find modules
export PYTHONPATH="/afs/cs.stanford.edu/u/suzeva/emergent-doordash/src:${PYTHONPATH:-}"