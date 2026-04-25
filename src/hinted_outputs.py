from __future__ import annotations

from typing import Any


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def extract_provider_reasoning(row: dict[str, Any]) -> str:
    """Return provider-supplied reasoning traces for hinted inference rows."""
    metadata = _metadata(row)
    for source in (row, metadata):
        for key in ("provider_reasoning", "reasoning_text", "reasoning", "reasoning_content"):
            text = _clean_text(source.get(key))
            if text:
                return text
    return ""


def extract_visible_model_output(row: dict[str, Any]) -> str:
    return _clean_text(row.get("model_output"))


def combined_model_response_text(row: dict[str, Any], *, include_labels: bool = False) -> str:
    reasoning = extract_provider_reasoning(row)
    visible_output = extract_visible_model_output(row)
    if not reasoning:
        return visible_output
    if not visible_output:
        return reasoning
    if include_labels:
        return f"Provider reasoning:\n{reasoning}\n\nVisible output:\n{visible_output}"
    return f"{reasoning}\n\n{visible_output}"


def response_text_stats(row: dict[str, Any]) -> dict[str, int]:
    reasoning = extract_provider_reasoning(row)
    visible_output = extract_visible_model_output(row)
    combined = combined_model_response_text(row)
    return {
        "provider_reasoning_chars": len(reasoning),
        "visible_output_chars": len(visible_output),
        "combined_output_chars": len(combined),
    }
