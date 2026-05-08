from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


# Local tokenizer counts can miss part of chat/request framing that vLLM
# enforces at serving time. Keep a buffer below the advertised context window.
MAX_TOKEN_CLAMP_SAFETY_MARGIN = 256


@dataclass(frozen=True)
class PromptTokenStats:
    context_limit: int
    prompt_token_counts: dict[str, int]


def extract_allowed_max_tokens_from_error(error_text: str) -> int | None:
    max_total_tokens_match = re.search(
        r"max_tokens\s*=\s*\d+\s*cannot\s+be\s+greater\s+than\s+"
        r"(?:max_model_len\s*=\s*)?max_total_tokens\s*=\s*(\d+)",
        error_text,
        flags=re.IGNORECASE,
    )
    if max_total_tokens_match is not None:
        return max(1, int(max_total_tokens_match.group(1)))

    paren_match = re.search(
        r"\((\d+)\s*>\s*(\d+)\s*-\s*(\d+)\)",
        error_text,
        flags=re.IGNORECASE,
    )
    if paren_match is not None:
        _, context_limit, input_tokens = (int(group) for group in paren_match.groups())
        return max(1, context_limit - input_tokens)

    request_input_match = re.search(
        r"maximum context length is\s*(\d+)(?:\s*tokens?)?\s*and your request has\s*"
        r"(\d+)\s*input tokens",
        error_text,
        flags=re.IGNORECASE,
    )
    if request_input_match is not None:
        context_limit, input_tokens = (int(group) for group in request_input_match.groups())
        return max(1, context_limit - input_tokens)

    qwen25_match = re.search(
        r"maximum context length is\s*(\d+)(?:\s*tokens?)?\.\s*however,\s*you requested\s*"
        r"\d+\s*output tokens?\s*and your prompt contains at least\s*(\d+)\s*input tokens",
        error_text,
        flags=re.IGNORECASE,
    )
    if qwen25_match is not None:
        context_limit, input_tokens = (int(group) for group in qwen25_match.groups())
        return max(1, context_limit - input_tokens)

    return None


def extract_context_limit_from_error(error_text: str) -> int | None:
    max_model_len_match = re.search(
        r"max_model_len\s*=\s*(\d+)",
        error_text,
        flags=re.IGNORECASE,
    )
    if max_model_len_match is not None:
        return max(1, int(max_model_len_match.group(1)))

    maximum_context_match = re.search(
        r"maximum context length is\s*(\d+)(?:\s*tokens?)?",
        error_text,
        flags=re.IGNORECASE,
    )
    if maximum_context_match is not None:
        return max(1, int(maximum_context_match.group(1)))

    paren_match = re.search(
        r"\((\d+)\s*>\s*(\d+)\s*-\s*(\d+)\)",
        error_text,
        flags=re.IGNORECASE,
    )
    if paren_match is not None:
        return max(1, int(paren_match.group(2)))

    return None


def apply_max_token_safety_margin(
    allowed_max_tokens: int,
    *,
    safety_margin: int = MAX_TOKEN_CLAMP_SAFETY_MARGIN,
) -> int:
    return max(1, allowed_max_tokens - max(0, safety_margin))


def format_exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return f"{type(exc).__name__}: {exc!r}"


def normalize_context_limit(raw_limit: Any, *, default: int) -> int:
    if isinstance(raw_limit, int) and 0 < raw_limit < 10_000_000:
        return raw_limit
    return default


def _token_ids_length(token_ids: Any) -> int | None:
    if token_ids is None:
        return None

    shape = getattr(token_ids, "shape", None)
    if shape is not None:
        try:
            if len(shape) > 0:
                return int(shape[-1])
        except Exception:
            pass

    ids = getattr(token_ids, "ids", None)
    if ids is not None:
        return _token_ids_length(ids)

    if isinstance(token_ids, (list, tuple)):
        if not token_ids:
            return 0
        first = token_ids[0]
        nested_length = _token_ids_length(first)
        if nested_length is not None and not isinstance(first, int):
            return nested_length
        return len(token_ids)

    try:
        return len(token_ids)
    except TypeError:
        return None


def _tokenized_length(tokenized: Any) -> int:
    input_ids = None
    if isinstance(tokenized, Mapping):
        input_ids = tokenized.get("input_ids")
    if input_ids is None:
        input_ids = getattr(tokenized, "input_ids", None)
    if input_ids is None:
        try:
            input_ids = tokenized["input_ids"]
        except Exception:
            input_ids = None

    length = _token_ids_length(input_ids)
    if length is not None:
        return length

    ids = getattr(tokenized, "ids", None)
    length = _token_ids_length(ids)
    if length is not None:
        return length

    encodings = getattr(tokenized, "encodings", None)
    if encodings:
        length = _token_ids_length(encodings[0])
        if length is not None:
            return length

    return len(tokenized)


def count_prompt_tokens_with_tokenizer(
    tokenizer: Any,
    *,
    messages: list[dict[str, str]] | None = None,
    prompt_text: str | None = None,
) -> int:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if messages and callable(apply_chat_template):
        token_ids = apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return _tokenized_length(token_ids)

    text = prompt_text or ""
    if not text and messages:
        text = "\n\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
    return len(tokenizer.encode(text, add_special_tokens=True))
