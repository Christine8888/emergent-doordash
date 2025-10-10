"""Example data structure for few-shot and prefill utilities."""

from dataclasses import dataclass


@dataclass
class Example:
    """Example data containing user and assistant messages.

    Args:
        question: The user's message (e.g., formatted question with choices)
        response: The assistant's response
        target: The target answer (e.g., "A", "B", "C", "D") - optional, used for prefill
    """
    question: str
    response: str
    target: str | None = None
