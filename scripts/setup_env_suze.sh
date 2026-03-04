#!/bin/bash
# Setup script for SLURM jobs - suzeva's version

# ----------------------------
# Constants (keep at top)
# ----------------------------
CONDA_BASE="/sphinx/u/${USER}/miniconda3"
CONDA_ENV_NAME="ed"
# Resolve repo root from this script location to support different clone paths.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ----------------------------
# Environment
# ----------------------------
# Keep everything in your own dirs (avoid any /.../cye/... paths).
# Mirrors the minimal style of scripts/setup_env.sh.
export HOME="/scr/${USER}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/sphinx/u/${USER}/models}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/sphinx/u/${USER}/.cache/huggingface/datasets}"

mkdir -p "$HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# Activate your conda env if available; otherwise fall back to your own venv.
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
else
    echo "ERROR: conda not found at $CONDA_BASE."
    echo "  If you want venv fallback, add it back here."
    exit 1
fi

# Add project src/ to PYTHONPATH so submitit workers can find modules
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"