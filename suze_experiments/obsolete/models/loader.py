"""Utilities for loading Hugging Face causal language models and tokenizers.

This module centralizes model/tokenizer loading for experiments and evaluations.
"""

from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


# Constants
DEFAULT_DTYPE: str = "bfloat16"
DEFAULT_DEVICE: str = "cuda"  # one of {"cuda", "cpu", "auto"}
DEFAULT_TRUST_REMOTE_CODE: bool = True
DEFAULT_PADDING_SIDE: str = "right"


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    """Map string dtype to torch dtype.

    Allowed values: "bfloat16", "float16", "float32".
    """
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    return torch.bfloat16


def load_causal_lm(
    model_id: str,
    *,
    revision: Optional[str] = None,
    dtype: str = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    trust_remote_code: bool = DEFAULT_TRUST_REMOTE_CODE,
    padding_side: str = DEFAULT_PADDING_SIDE,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a HF causal LM and tokenizer with consistent defaults.

    Args:
        model_id: Hugging Face model repo id (e.g., "openlm-research/open_llama_3b").
        revision: Optional HF revision (branch, tag, or commit sha).
        dtype: One of {"bfloat16", "float16", "float32"}.
        device: One of {"cuda", "cpu", "auto"}. If "auto", lets HF shard across devices.
        trust_remote_code: Whether to allow custom model/tokenizer code.
        padding_side: Tokenizer padding side ("right" for loss computation; "left" for generation).

    Returns:
        (model, tokenizer)
    """
    torch_dtype = _resolve_torch_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    tokenizer.padding_side = padding_side

    # Ensure pad_token is set when absent to avoid DataCollator issues during loss eval
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if device == "auto" else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        revision=revision,
        use_safetensors=True,
    )

    if device_map is None:
        if device == "cuda":
            model.to("cuda")
        elif device == "cpu":
            model.to("cpu")

    model.eval()
    return model, tokenizer


