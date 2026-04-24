from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.storage import write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_BACKUP_ROOT = PROJECT_ROOT / "data" / "backups" / "hinted_inference_is_error_cleanup"
ALLOWED_MODELS = {
    "Llama-2-7b-chat-hf",
    "Qwen3-32B",
}


@dataclass(frozen=True)
class FileCleanupPlan:
    path: Path
    model: str
    combo: str
    hint_fraction: str
    total_rows: int
    removed_rows: int
    kept_rows: int
    error_inference_ids: list[str]


def _utcnow_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_models(value: str) -> list[str]:
    models = [part.strip() for part in value.split(",") if part.strip()]
    if not models:
        raise ValueError("models cannot be empty")
    invalid = sorted(set(models) - ALLOWED_MODELS)
    if invalid:
        raise ValueError(
            f"unsupported models requested: {invalid}. Allowed models: {sorted(ALLOWED_MODELS)}"
        )
    return sorted(set(models))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove persisted is_error=true rows from hinted inference JSONL files for a small "
            "allowlisted set of models. Dry-run by default."
        )
    )
    parser.add_argument("--benchmark", type=str, default="aime2025_2026")
    parser.add_argument("--hint-type", type=str, default="answer_not_revealed")
    parser.add_argument("--fractioner", type=str, default="mask_word")
    parser.add_argument(
        "--models",
        type=_parse_models,
        default=sorted(ALLOWED_MODELS),
        help=(
            "Comma-separated subset of allowed models. "
            f"Allowed: {', '.join(sorted(ALLOWED_MODELS))}"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually back up and rewrite files. Without this flag, only print the plan.",
    )
    return parser


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _discover_candidate_files(
    *,
    data_root: Path,
    benchmark: str,
    hint_type: str,
    fractioner: str,
    models: list[str],
) -> list[Path]:
    files: list[Path] = []
    combo = f"{hint_type}__{fractioner}"
    for model in models:
        model_dir = data_root / "hinted_inference" / benchmark / model / combo
        if not model_dir.exists():
            continue
        files.extend(sorted(model_dir.glob("fraction_*.jsonl")))
    return sorted(files)


def _build_cleanup_plan(
    *,
    data_root: Path,
    benchmark: str,
    hint_type: str,
    fractioner: str,
    models: list[str],
) -> list[FileCleanupPlan]:
    plans: list[FileCleanupPlan] = []
    for path in _discover_candidate_files(
        data_root=data_root,
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
        models=models,
    ):
        rows = _load_jsonl_rows(path)
        removed_rows = 0
        error_inference_ids: list[str] = []
        for row in rows:
            if row.get("is_error") is True:
                removed_rows += 1
                inference_id = row.get("inference_id")
                if isinstance(inference_id, str) and inference_id:
                    error_inference_ids.append(inference_id)
        if removed_rows == 0:
            continue
        plans.append(
            FileCleanupPlan(
                path=path,
                model=path.parts[-3],
                combo=path.parts[-2],
                hint_fraction=path.name.removeprefix("fraction_").removesuffix(".jsonl"),
                total_rows=len(rows),
                removed_rows=removed_rows,
                kept_rows=len(rows) - removed_rows,
                error_inference_ids=sorted(error_inference_ids),
            )
        )
    return plans


def _manifest_payload(
    *,
    args: argparse.Namespace,
    plans: list[FileCleanupPlan],
    backup_dir: Path | None,
    applied: bool,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "applied": applied,
        "benchmark": args.benchmark,
        "hint_type": args.hint_type,
        "fractioner": args.fractioner,
        "models": list(args.models),
        "data_root": str(args.data_root),
        "backup_root": str(args.backup_root),
        "backup_dir": str(backup_dir) if backup_dir is not None else None,
        "files": [
            {
                "path": str(plan.path),
                "model": plan.model,
                "combo": plan.combo,
                "hint_fraction": plan.hint_fraction,
                "total_rows": plan.total_rows,
                "removed_rows": plan.removed_rows,
                "kept_rows": plan.kept_rows,
                "error_inference_ids": plan.error_inference_ids,
            }
            for plan in plans
        ],
    }


def _apply_cleanup(
    *,
    plans: list[FileCleanupPlan],
    data_root: Path,
    backup_root: Path,
    args: argparse.Namespace,
) -> Path:
    timestamp = _utcnow_compact()
    backup_dir = backup_root / timestamp
    manifest_dir = backup_dir / "_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=False)

    for plan in plans:
        relative_path = plan.path.relative_to(data_root)
        backup_path = backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan.path, backup_path)

    manifest_path = manifest_dir / "cleanup_manifest.json"
    manifest_payload = _manifest_payload(
        args=args,
        plans=plans,
        backup_dir=backup_dir,
        applied=True,
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    for plan in plans:
        rows = _load_jsonl_rows(plan.path)
        filtered_rows = [row for row in rows if row.get("is_error") is not True]
        if len(filtered_rows) != plan.kept_rows:
            raise RuntimeError(
                f"filtered row count mismatch for {plan.path}: "
                f"expected {plan.kept_rows}, got {len(filtered_rows)}"
            )
        write_jsonl(plan.path, filtered_rows)

    return backup_dir


def _print_summary(*, plans: list[FileCleanupPlan], backup_dir: Path | None, applied: bool) -> None:
    total_removed = sum(plan.removed_rows for plan in plans)
    total_kept = sum(plan.kept_rows for plan in plans)
    print("[cleanup_hinted_is_error_rows] summary")
    print(f"  applied={applied}")
    print(f"  affected_files={len(plans)}")
    print(f"  removed_rows={total_removed}")
    print(f"  kept_rows={total_kept}")
    if backup_dir is not None:
        print(f"  backup_dir={backup_dir}")
    for plan in plans:
        print(
            f"  {plan.model} {plan.combo} fraction={plan.hint_fraction} "
            f"remove={plan.removed_rows} keep={plan.kept_rows} path={plan.path}"
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    plans = _build_cleanup_plan(
        data_root=args.data_root,
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
        models=args.models,
    )
    if not plans:
        print("[cleanup_hinted_is_error_rows] no affected files found")
        return

    if args.apply:
        backup_dir = _apply_cleanup(
            plans=plans,
            data_root=args.data_root,
            backup_root=args.backup_root,
            args=args,
        )
        _print_summary(plans=plans, backup_dir=backup_dir, applied=True)
        return

    manifest_payload = _manifest_payload(
        args=args,
        plans=plans,
        backup_dir=None,
        applied=False,
    )
    print("[cleanup_hinted_is_error_rows] dry-run manifest")
    print(json.dumps(manifest_payload, indent=2, sort_keys=True))
    _print_summary(plans=plans, backup_dir=None, applied=False)


if __name__ == "__main__":
    # python -m runs.cleanup_hinted_is_error_rows
    # python -m runs.cleanup_hinted_is_error_rows --models Llama-2-7b-chat-hf,Qwen3-32B --apply
    main()
