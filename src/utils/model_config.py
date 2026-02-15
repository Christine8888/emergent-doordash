"""Model-specific configuration."""

from dataclasses import dataclass
from typing import Any

# Explicit node lists by GPU capability
SMALL_MODEL_NODES = "sphinx[1-11],miso[1-5]" #,jagupard[32-39]"  # A100/H100/H200 + A6000 - for ≤4B
LARGE_MODEL_NODES = "sphinx[1-11],miso[1-5]"  # A100/H100/H200 only - for 8B-14B
H200_NODES = "sphinx[10-11],miso[1-5]"  # H200 only - for 32B+ with TP=1

# Partition configurations (must include all partitions that contain the nodes above)
SMALL_MODEL_PARTITIONS = "sphinx,miso,jag-standard"
LARGE_MODEL_PARTITIONS = "sphinx,miso"

MODEL_PREFILL_TOKENS = {
    "Qwen3": "<think>",
}

# Generation config defaults per model family
# Qwen3 recommended: temperature=0.6, top_p=0.95, top_k=20 for thinking mode
# presence_penalty helps prevent endless repetitions in thinking mode
# See: https://huggingface.co/Qwen/Qwen3-32B, https://qwen.readthedocs.io/en/latest/deployment/vllm.html
MODEL_GENERATION_DEFAULTS: dict[str, dict[str, Any]] = {
    "Qwen3": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.0},
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


# Qwen3 models
QWEN3_MODELS = [
    ModelSpec("Qwen/Qwen3-0.6B"),
    ModelSpec("Qwen/Qwen3-1.7B"),
    ModelSpec("Qwen/Qwen3-4B"),
    ModelSpec("Qwen/Qwen3-8B", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES),
    ModelSpec("Qwen/Qwen3-14B", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES),
    ModelSpec("Qwen/Qwen3-32B", partitions=LARGE_MODEL_PARTITIONS, nodelist=H200_NODES),
]

# Qwen2.5 models
QWEN25_MODELS = [
    # ModelSpec("Qwen/Qwen2.5-0.5B-Instruct"), # doesn't run inference well
    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct"),
    ModelSpec("Qwen/Qwen2.5-3B-Instruct"),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES),
    ModelSpec("Qwen/Qwen2.5-14B-Instruct", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES),
    ModelSpec("Qwen/Qwen2.5-32B-Instruct", partitions=LARGE_MODEL_PARTITIONS, nodelist=H200_NODES),
]

# Llama models
LLAMA_MODELS = [
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES),
    ModelSpec("meta-llama/Llama-3.1-70B-Instruct", tp=2, partitions=LARGE_MODEL_PARTITIONS, nodelist=H200_NODES),
]

# Gemma models
GEMMA_MODELS = [
    # ModelSpec("google/gemma-3-1b-it"), # doesn't run inference well
    ModelSpec("google/gemma-3-4b-it"),
    ModelSpec("google/gemma-3-12b-it", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES),
    ModelSpec("google/gemma-3-27b-it", tp=2, partitions=LARGE_MODEL_PARTITIONS,
  nodelist=LARGE_MODEL_NODES),
]
