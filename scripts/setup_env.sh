#!/bin/bash
# Setup script for SLURM jobs - creates or activates venv

export SCR=/juice5b/scr5b/cye
export HOME="$SCR"
export VENV_DIR="$SCR/.venv"
export UV_CACHE_DIR="$SCR/.cache/uv"
export HF_HOME="/sphinx/u/cye/.cache/huggingface"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    mkdir -p "$SCR"
    cd "$SCR"

    uv python install 3.11
    uv venv --python 3.11
    source .venv/bin/activate

    uv pip install ipykernel
    uv run python -m ipykernel install --user --name "venv"
    uv pip install dotenv trl transformers torch "huggingface_hub[cli]" wandb dotenv deepspeed
    uv pip install vllm --torch-backend=auto
    cd /sphinx/u/cye/emergent-doordash
    uv pip install -e .
    uv pip install inspect-ai
    uv pip install --upgrade openai anthropic transformers
    uv pip install numpy==2.2
fi
