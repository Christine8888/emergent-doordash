from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import math
import random
import re
from typing import Any

from src.hint_types import get_hint_type_spec
from src.types import HintGenerationRecord


def _deterministic_rng(*parts: object) -> random.Random:
    joined = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    return random.Random(seed)


def _visible_count(total_units: int, hint_fraction: float) -> int:
    if total_units <= 0:
        return 0
    keep = int(math.ceil(total_units * hint_fraction))
    if keep < 0:
        keep = 0
    if keep > total_units:
        keep = total_units
    return keep


def _normalize_fraction(hint_fraction: float) -> float:
    if hint_fraction < 0.0 or hint_fraction > 1.0:
        raise ValueError("hint_fraction must be in [0.0, 1.0]")
    return hint_fraction


def _parse_bag_hints(text: str) -> list[str]:
    pattern = re.compile(
        r"<hint\s+id\s*=\s*['\"]?(\d+)['\"]?\s*>(.*?)</hint>",
        re.IGNORECASE | re.DOTALL,
    )
    matches = pattern.findall(text)
    if not matches:
        return []

    by_id: dict[int, str] = {}
    for raw_id, hint_text in matches:
        idx = int(raw_id)
        by_id[idx] = hint_text.strip()
    return [by_id[i] for i in sorted(by_id.keys())]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?](?=\s|$))?", text):
        segment = match.group(0)
        if segment.strip():
            spans.append((match.start(), match.end()))
    return spans


class HintFractionerSpec(ABC):
    name: str

    @abstractmethod
    def apply(
        self,
        *,
        hint_record: HintGenerationRecord,
        hint_fraction: float,
    ) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError


class BagCountFractioner(HintFractionerSpec):
    name = "bag_count"

    def apply(
        self,
        *,
        hint_record: HintGenerationRecord,
        hint_fraction: float,
    ) -> tuple[str, dict[str, Any]]:
        _normalize_fraction(hint_fraction)
        hints = _parse_bag_hints(hint_record.full_hint)
        keep = _visible_count(total_units=10, hint_fraction=hint_fraction)
        shown_hints = hints[:keep]
        text = "\n".join(shown_hints)
        metadata = {
            "fractioner": self.name,
            "unit_mode": "hint",
            "units_total": 10,
            "units_visible": keep,
            "units_masked": 10 - keep,
        }
        return text, metadata


class TruncateSentenceFractioner(HintFractionerSpec):
    name = "truncate_sentence"

    def apply(
        self,
        *,
        hint_record: HintGenerationRecord,
        hint_fraction: float,
    ) -> tuple[str, dict[str, Any]]:
        _normalize_fraction(hint_fraction)
        spans = _sentence_spans(hint_record.full_hint)
        total = len(spans)
        keep = _visible_count(total_units=total, hint_fraction=hint_fraction)
        if keep == 0:
            text = ""
        elif keep >= total:
            text = hint_record.full_hint
        else:
            _, end = spans[keep - 1]
            text = hint_record.full_hint[:end]
        metadata = {
            "fractioner": self.name,
            "unit_mode": "sentence",
            "units_total": total,
            "units_visible": keep,
            "units_masked": total - keep,
        }
        return text, metadata


class TruncateWordFractioner(HintFractionerSpec):
    name = "truncate_word"

    def apply(
        self,
        *,
        hint_record: HintGenerationRecord,
        hint_fraction: float,
    ) -> tuple[str, dict[str, Any]]:
        _normalize_fraction(hint_fraction)
        matches = list(re.finditer(r"\S+", hint_record.full_hint))
        total = len(matches)
        keep = _visible_count(total_units=total, hint_fraction=hint_fraction)
        if keep == 0:
            text = ""
        elif keep >= total:
            text = hint_record.full_hint
        else:
            end = matches[keep - 1].end()
            text = hint_record.full_hint[:end]
        metadata = {
            "fractioner": self.name,
            "unit_mode": "word",
            "units_total": total,
            "units_visible": keep,
            "units_masked": total - keep,
        }
        return text, metadata


class MaskSentenceFractioner(HintFractionerSpec):
    name = "mask_sentence"

    def apply(
        self,
        *,
        hint_record: HintGenerationRecord,
        hint_fraction: float,
    ) -> tuple[str, dict[str, Any]]:
        _normalize_fraction(hint_fraction)
        spans = _sentence_spans(hint_record.full_hint)
        total = len(spans)
        keep = _visible_count(total_units=total, hint_fraction=hint_fraction)
        visible_idxs = set(range(total))
        if total > keep:
            rng = _deterministic_rng(hint_record.hint_id, self.name, hint_fraction)
            mask_idxs = set(rng.sample(range(total), total - keep))
            visible_idxs = set(range(total)) - mask_idxs

        out_parts: list[str] = []
        cursor = 0
        for idx, (start, end) in enumerate(spans):
            out_parts.append(hint_record.full_hint[cursor:start])
            if idx in visible_idxs:
                out_parts.append(hint_record.full_hint[start:end])
            else:
                out_parts.append("[MASK]")
            cursor = end
        out_parts.append(hint_record.full_hint[cursor:])
        text = "".join(out_parts)
        metadata = {
            "fractioner": self.name,
            "unit_mode": "sentence",
            "units_total": total,
            "units_visible": keep,
            "units_masked": total - keep,
        }
        return text, metadata


class MaskWordFractioner(HintFractionerSpec):
    name = "mask_word"

    def apply(
        self,
        *,
        hint_record: HintGenerationRecord,
        hint_fraction: float,
    ) -> tuple[str, dict[str, Any]]:
        _normalize_fraction(hint_fraction)
        matches = list(re.finditer(r"\S+", hint_record.full_hint))
        total = len(matches)
        keep = _visible_count(total_units=total, hint_fraction=hint_fraction)

        visible_idxs = set(range(total))
        if total > keep:
            rng = _deterministic_rng(hint_record.hint_id, self.name, hint_fraction)
            mask_idxs = set(rng.sample(range(total), total - keep))
            visible_idxs = set(range(total)) - mask_idxs

        out_parts: list[str] = []
        cursor = 0
        for idx, match in enumerate(matches):
            out_parts.append(hint_record.full_hint[cursor:match.start()])
            if idx in visible_idxs:
                out_parts.append(match.group(0))
            else:
                out_parts.append("[MASK]")
            cursor = match.end()
        out_parts.append(hint_record.full_hint[cursor:])

        text = "".join(out_parts)
        metadata = {
            "fractioner": self.name,
            "unit_mode": "word",
            "units_total": total,
            "units_visible": keep,
            "units_masked": total - keep,
        }
        return text, metadata


FRACTIONER_SPECS: dict[str, HintFractionerSpec] = {
    "bag_count": BagCountFractioner(),
    "truncate_sentence": TruncateSentenceFractioner(),
    "truncate_word": TruncateWordFractioner(),
    "mask_sentence": MaskSentenceFractioner(),
    "mask_word": MaskWordFractioner(),
}


def get_hint_fractioner_spec(name: str) -> HintFractionerSpec:
    return FRACTIONER_SPECS[name]


def fraction_hint(
    *,
    hint_record: HintGenerationRecord,
    fractioner_name: str,
    hint_fraction: float,
) -> tuple[str, dict[str, Any]]:
    hint_type_spec = get_hint_type_spec(hint_record.hint_type)
    if fractioner_name not in hint_type_spec.allowed_fractioners:
        raise ValueError(
            f"Invalid fractioner {fractioner_name!r} for hint_type={hint_record.hint_type!r}. "
            f"Allowed fractioners: {list(hint_type_spec.allowed_fractioners)}"
        )

    fractioner = get_hint_fractioner_spec(fractioner_name)
    return fractioner.apply(hint_record=hint_record, hint_fraction=hint_fraction)
