"""Stable identifiers for corpus tables."""

from __future__ import annotations

import hashlib
from typing import Any


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def make_eval_id(source_owner: str, eval_rel_path: str) -> str:
    return stable_hash(f"{source_owner}|{eval_rel_path}")


def make_rollout_id(
    eval_id: str,
    rollout_ordinal: int,
    sample_id: Any,
    epoch: Any,
    sample_idx: Any,
) -> str:
    return stable_hash(f"{eval_id}|{rollout_ordinal}|{sample_id}|{epoch}|{sample_idx}")


def make_grader_result_id(
    rollout_id: str,
    grader_origin: str,
    grader_name: str,
    grader_version: str,
) -> str:
    return stable_hash(f"{rollout_id}|{grader_origin}|{grader_name}|{grader_version}")

