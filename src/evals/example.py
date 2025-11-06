"""Example data structure for few-shot and prefill utilities."""

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Example:
    """Example data containing user and assistant messages.

    Standardized fields that must be present in JSONL files:
    - id: Sample identifier (required)
    - question: The user's message/prompt (required)
    - response: The assistant's response (optional, for few-shot)
    - target: The target answer (optional, for validation)

    Args:
        id: Sample identifier
        question: The user's message (e.g., formatted question with choices)
        response: The assistant's response (optional)
        target: The target answer (e.g., "A", "B", "C", "D") (optional)
    """
    id: str
    question: str
    response: str | None = None
    target: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Example":
        """Create Example from dictionary.

        Args:
            data: Dictionary with keys: id, question, response (optional), target (optional)

        Returns:
            Example instance

        Raises:
            KeyError: If required fields (id, question) are missing
            ValueError: If data is invalid
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")

        required_fields = ["id", "question"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise KeyError(f"Missing required fields: {missing}")

        return cls(
            id=data["id"],
            question=data["question"],
            response=data.get("response"),
            target=data.get("target"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert Example to dictionary.

        Returns:
            Dictionary with all fields, including None values
        """
        return asdict(self)
