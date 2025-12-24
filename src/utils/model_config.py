"""Model-specific configuration."""

from dataclasses import dataclass

# Explicit node lists by GPU capability
SMALL_MODEL_NODES = "sphinx[1-11],miso[1-5],jagupard[32-39]"  # A100/H100/H200 + A6000 - for ≤8B
LARGE_MODEL_NODES = "sphinx[1-11],miso[1-5]"  # A100/H100/H200 only - for 12B-14B
H200_NODES = "sphinx[10-11],miso[1-5]"  # H200 only - for 32B+ with TP=1

# Partition configurations (must include all partitions that contain the nodes above)
SMALL_MODEL_PARTITIONS = "sphinx,miso,jag-standard"
LARGE_MODEL_PARTITIONS = "sphinx,miso"

MODEL_PREFILL_TOKENS = {
    "Qwen3": "<think>",
}


def get_start_prefill(model_name: str) -> str | None:
    """Get start prefill token for model family."""
    for prefix, token in MODEL_PREFILL_TOKENS.items():
        if prefix in model_name:
            return token
    return None


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
    ModelSpec("Qwen/Qwen3-8B"),
    ModelSpec("Qwen/Qwen3-14B", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES),
    ModelSpec("Qwen/Qwen3-32B", partitions=LARGE_MODEL_PARTITIONS, nodelist=H200_NODES),
]

# Qwen2.5 models
QWEN25_MODELS = [
    ModelSpec("Qwen/Qwen2.5-0.5B-Instruct"),
    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct"),
    ModelSpec("Qwen/Qwen2.5-3B-Instruct"),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct"),
    ModelSpec("Qwen/Qwen2.5-14B-Instruct", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES),
    ModelSpec("Qwen/Qwen2.5-32B-Instruct", partitions=LARGE_MODEL_PARTITIONS, nodelist=H200_NODES),
]

# Llama models
LLAMA_MODELS = [
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct"),
    ModelSpec("meta-llama/Llama-3.1-70B-Instruct", tp=2, partitions=LARGE_MODEL_PARTITIONS, nodelist=H200_NODES),
]

# Gemma models
GEMMA_MODELS = [
    ModelSpec("google/gemma-3-4b-it"),
    ModelSpec("google/gemma-3-12b-it", partitions=LARGE_MODEL_PARTITIONS, nodelist=LARGE_MODEL_NODES),
    ModelSpec("google/gemma-3-27b-it", partitions=LARGE_MODEL_PARTITIONS, nodelist=H200_NODES),
]