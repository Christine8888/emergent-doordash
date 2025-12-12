"""Model-specific configuration."""

MODEL_PREFILL_TOKENS = {
    "Qwen3": "<think>",
}


def get_start_prefill(model_name: str) -> str | None:
    """Get start prefill token for model family."""
    for prefix, token in MODEL_PREFILL_TOKENS.items():
        if prefix in model_name:
            return token
    return None
