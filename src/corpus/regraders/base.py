"""Base types for corpus regraders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class RegraderSpec:
    name: str
    version: str
    benchmark_slugs: tuple[str, ...]
    applies: Callable[[dict[str, Any]], bool]
    run: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
