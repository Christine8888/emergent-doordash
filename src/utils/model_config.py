"""Model-specific configuration."""

from dataclasses import dataclass
from typing import Any

# Explicit node lists by GPU capability
SMALL_MODEL_NODES = "sphinx[1-11],miso[1-5],jagupard[32-39]"  # A100/H100/H200 + A6000 - for ≤4B
LARGE_MODEL_NODES = "sphinx[1-11],miso[1-5]"  # A100/H100/H200 only - for 8B-14B
H200_NODES = "sphinx[10-11],miso[1-5]"  # H200 only - for 32B+ with TP=1

# Partition configurations (must include all partitions that contain the nodes above)
SMALL_MODEL_PARTITIONS = "sphinx,miso,jag-standard"
LARGE_MODEL_PARTITIONS = "sphinx,miso"

# sc-loprio (scavenger) scheduling: partition and SLURM --constraint values by model size.
# Node features: 40G=A100-40GB, 48G=A6000/A40/L40S, 80G=A100-80GB/H100, 141G=H200.
# Usable VRAM at gpu_memory_utilization=0.85: 40G→34GB, 48G→40.8GB, 80G→68GB, 141G→120GB.
SC_LOPRIO_PARTITION = "sc-loprio"
SMALL_MODEL_CONSTRAINT  = "80G|141G|40G|48G"  # ≤4B:  weights ≤8GB,  fits anywhere
MEDIUM_MODEL_CONSTRAINT = "80G|141G|48G"       # 7–27B (tp≥1): weights ≤27GB/GPU, fits on 48G+
LARGE_MODEL_CONSTRAINT  = "80G|141G"           # 14B:  weights ~28GB, 48G too tight at 32K ctx
H200_CONSTRAINT         = "141G"               # 32B/70B: weights exceed 80G usable VRAM

MODEL_PREFILL_TOKENS = {
    "Qwen3": "<think>",
}

# Generation config defaults per model family
# Qwen3 recommended: temperature=0.6, top_p=0.95, top_k=20 for thinking mode
# See: https://huggingface.co/Qwen/Qwen3-32B, https://qwen.readthedocs.io/en/latest/deployment/vllm.html
MODEL_GENERATION_DEFAULTS: dict[str, dict[str, Any]] = {
    "Qwen3": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
}

# Default for models without specific config (standard sampling)
DEFAULT_GENERATION_CONFIG: dict[str, Any] = {"temperature": 1.0}


def get_start_prefill(model_name: str) -> str | None:
    """Get start prefill token for model family."""
    for prefix, token in MODEL_PREFILL_TOKENS.items():
        if prefix in model_name:
            return token
    return None


def get_generation_defaults(model_name: str) -> dict[str, Any]:
    """Get generation config defaults for model family.

    Args:
        model_name: Model name (e.g., "Qwen3-32B", "Llama-3.1-8B-Instruct")

    Returns:
        Dict of generation parameters (temperature, top_p, top_k, etc.)
    """
    for prefix, config in MODEL_GENERATION_DEFAULTS.items():
        if prefix in model_name:
            return config
    return DEFAULT_GENERATION_CONFIG


@dataclass
class ModelSpec:
    """Model specification with SLURM routing info."""
    path: str
    tp: int = 1
    partitions: str = SMALL_MODEL_PARTITIONS
    nodelist: str = SMALL_MODEL_NODES
    account: str | None = None
    constraint: str | None = None  # SLURM --constraint, used for sc-loprio scheduling


# Qwen3 models
QWEN3_MODELS = [
    ModelSpec("Qwen/Qwen3-0.6B", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES, constraint=SMALL_MODEL_CONSTRAINT),
    ModelSpec("Qwen/Qwen3-1.7B", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES, constraint=SMALL_MODEL_CONSTRAINT),
    ModelSpec("Qwen/Qwen3-4B", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES, constraint=SMALL_MODEL_CONSTRAINT),
    ModelSpec("Qwen/Qwen3-8B", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES, constraint=MEDIUM_MODEL_CONSTRAINT),
    ModelSpec("Qwen/Qwen3-14B", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES, constraint=LARGE_MODEL_CONSTRAINT),
    ModelSpec("Qwen/Qwen3-32B", partitions=LARGE_MODEL_PARTITIONS, nodelist=H200_NODES, constraint=H200_CONSTRAINT),
]

# Qwen2.5 models
QWEN25_MODELS = [
    # ModelSpec("Qwen/Qwen2.5-0.5B-Instruct"), # doesn't run inference well
    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES, constraint=SMALL_MODEL_CONSTRAINT),
    ModelSpec("Qwen/Qwen2.5-3B-Instruct", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES, constraint=SMALL_MODEL_CONSTRAINT),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES, constraint=MEDIUM_MODEL_CONSTRAINT),
    ModelSpec("Qwen/Qwen2.5-14B-Instruct", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES, constraint=LARGE_MODEL_CONSTRAINT),
    ModelSpec("Qwen/Qwen2.5-32B-Instruct", partitions=LARGE_MODEL_PARTITIONS, nodelist=H200_NODES, constraint=H200_CONSTRAINT),
]

# Llama models
LLAMA_MODELS = [
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES, constraint=MEDIUM_MODEL_CONSTRAINT),
    ModelSpec("meta-llama/Llama-3.1-70B-Instruct", tp=2, partitions=LARGE_MODEL_PARTITIONS, nodelist=H200_NODES, constraint=H200_CONSTRAINT),
]

# Gemma models
GEMMA_MODELS = [
    # ModelSpec("google/gemma-3-1b-it", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES, constraint=SMALL_MODEL_CONSTRAINT), # doesn't run inference well
    ModelSpec("google/gemma-3-4b-it", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES, constraint=SMALL_MODEL_CONSTRAINT),
    ModelSpec("google/gemma-3-12b-it", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES, constraint=MEDIUM_MODEL_CONSTRAINT),
    ModelSpec("google/gemma-3-27b-it", tp=2, partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES, constraint=MEDIUM_MODEL_CONSTRAINT),
]
