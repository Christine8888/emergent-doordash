"""Example data structure for few-shot and prefill utilities."""

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Example:
    """Example data containing user and assistant messages.

    Standardized fields that must be present in JSONL files:
    - id: Sample identifier (required)
    - question: The user's message (required)
    - target: The target answer (required)
    - response: The full response from the model (required)
    - hint: Hint data handled by solver, can be any type (required)
    - sample_idx: Index for multiple samples of same question (required, default 0)
    - prompt: The full prompt into the model (optional)
    - metadata: Additional metadata dict (optional)

    Args:
        id: Sample identifier
        question: The user's message (e.g., formatted question with choices)
        target: The target answer (e.g., "A", "B", "C", "D", or numeric answer)
        response: The full response from the model
        hint: Hint data for solver (any type)
        sample_idx: Index for multiple samples of same question (default 0)
        prompt: The full prompt into the model (optional)
        metadata: Additional metadata dict (optional)
    """
    id: str
    question: str
    target: str
    response: str
    hint: Any
    sample_idx: int = 0
    prompt: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Example":
        """Create Example from dictionary.

        Args:
            data: Dictionary with keys: id, question, target, response, hint (required),
                  sample_idx (optional, default 0), prompt (optional), metadata (optional)

        Returns:
            Example instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")

        required_fields = ["id", "question", "target", "response", "hint"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise KeyError(f"Missing required fields: {missing}")

        return cls(
            id=data["id"],
            question=data["question"],
            target=data["target"],
            response=data["response"],
            hint=data["hint"],
            sample_idx=data.get("sample_idx", 0),
            prompt=data.get("prompt"),
            metadata=data.get("metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert Example to dictionary.

        Returns:
            Dictionary with all fields, including None values
        """
        return asdict(self)

    def has_valid_hint(self) -> bool:
        """Check if this example has a valid hint.

        Returns:
            True if hint is a non-empty string.
        """
        return isinstance(self.hint, str) and bool(self.hint.strip())
