from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def make_stable_id(*parts: object, length: int) -> str:
    """Create a deterministic short ID from arbitrary parts."""
    joined = "||".join(str(part).strip() for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return digest[:length]


def _safe_component(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned or "unknown"


def build_hint_generation_path(
    benchmark_name: str,
    hint_type: str,
    *,
    data_root: str | Path,
) -> Path:
    benchmark = _safe_component(benchmark_name)
    hint = _safe_component(hint_type)
    return Path(data_root) / "hint_generation" / benchmark / f"{hint}.jsonl"


def build_hinted_inference_path(
    benchmark_name: str,
    model: str,
    hint_type: str,
    hint_fraction: float,
    *,
    data_root: str | Path,
) -> Path:
    benchmark = _safe_component(benchmark_name)
    model_name = _safe_component(model)
    hint = _safe_component(hint_type)
    fraction_text = f"{hint_fraction:.4f}".rstrip("0").rstrip(".")
    if not fraction_text:
        fraction_text = "0"
    return Path(data_root) / "hinted_inference" / benchmark / model_name / hint / f"fraction_{fraction_text}.jsonl"


def append_jsonl(path: str | Path, record: BaseModel | dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = record.model_dump() if isinstance(record, BaseModel) else record
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            payload = record.model_dump() if isinstance(record, BaseModel) else record
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")


def iter_jsonl(path: str | Path, model_cls: type[T] | None):
    path = Path(path)
    if not path.exists():
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if model_cls is None:
                yield item
            else:
                yield model_cls.model_validate(item)


def read_jsonl(path: str | Path, model_cls: type[T] | None) -> list[dict[str, Any] | T]:
    rows = iter_jsonl(path, model_cls=model_cls)
    if rows is None:
        return []
    return list(rows)
