"""Registry for available corpus regraders."""

from __future__ import annotations

from corpus.regraders.base import RegraderSpec
from corpus.regraders.math_extract_fixed_v1 import SPEC as MATH_EXTRACT_FIXED_V1


REGRADER_REGISTRY: dict[str, RegraderSpec] = {
    MATH_EXTRACT_FIXED_V1.name: MATH_EXTRACT_FIXED_V1,
}


def resolve_regraders(enabled_names: list[str]) -> list[RegraderSpec]:
    unknown = [name for name in enabled_names if name not in REGRADER_REGISTRY]
    if unknown:
        available = sorted(REGRADER_REGISTRY.keys())
        raise ValueError(
            f"Unknown regrader(s): {unknown}. Available: {available}"
        )
    return [REGRADER_REGISTRY[name] for name in enabled_names]

