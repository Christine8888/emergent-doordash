"""Model-specific configuration."""

from dataclasses import dataclass

# Node groups by GPU type
H200_NODES = "sphinx[10-11],miso[1-5]"

# Partition configurations
LARGE_GPU_PARTITIONS = "sphinx,miso"  # 80GB+ GPUs (A100, H100, H200) - for 12B+
ALL_PARTITIONS = "sphinx,miso,jag-standard"  # Includes 48GB GPUs (A6000) - for 8B and under

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
    partitions: str = ALL_PARTITIONS
    nodelist: str | None = None


# Qwen3 models
QWEN3_MODELS = [
    ModelSpec("Qwen/Qwen3-0.6B"),
    ModelSpec("Qwen/Qwen3-1.7B"),
    ModelSpec("Qwen/Qwen3-4B"),
    ModelSpec("Qwen/Qwen3-8B"),
    ModelSpec("Qwen/Qwen3-14B", partitions=LARGE_GPU_PARTITIONS),
    ModelSpec("Qwen/Qwen3-32B", partitions=LARGE_GPU_PARTITIONS, nodelist=H200_NODES),
]

# Qwen2.5 models
QWEN25_MODELS = [
    ModelSpec("Qwen/Qwen2.5-0.5B-Instruct"),
    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct"),
    ModelSpec("Qwen/Qwen2.5-3B-Instruct"),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct"),
    ModelSpec("Qwen/Qwen2.5-14B-Instruct", partitions=LARGE_GPU_PARTITIONS),
    ModelSpec("Qwen/Qwen2.5-32B-Instruct", partitions=LARGE_GPU_PARTITIONS, nodelist=H200_NODES),
]

# Llama models
LLAMA_MODELS = [
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct"),
    ModelSpec("meta-llama/Llama-3.1-70B-Instruct", tp=2, partitions=LARGE_GPU_PARTITIONS, nodelist=H200_NODES),
]

# Gemma models
GEMMA_MODELS = [
    ModelSpec("google/gemma-3-4b-it"),
    ModelSpec("google/gemma-3-12b-it", partitions=LARGE_GPU_PARTITIONS),
    ModelSpec("google/gemma-3-27b-it", partitions=LARGE_GPU_PARTITIONS, nodelist=H200_NODES),
]