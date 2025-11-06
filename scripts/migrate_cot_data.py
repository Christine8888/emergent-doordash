#!/usr/bin/env python3
"""
Migrate CoT data files to new Example format.

Transformations:
- full_prompt -> prompt
- response -> hint (keep response as well)
- question_with_choices -> question
- Remove: choices, task_id
- Keep all other fields
"""
import json
import sys
from pathlib import Path


def migrate_record(record: dict) -> dict:
    """Migrate a single record to new format."""
    migrated = {}

    # Required fields
    migrated["id"] = record["id"]
    migrated["target"] = record["target"]

    # Field mappings
    if "full_prompt" in record:
        migrated["prompt"] = record["full_prompt"]

    if "question_with_choices" in record:
        migrated["question"] = record["question_with_choices"]
    elif "question" in record:
        migrated["question"] = record["question"]

    if "response" in record:
        migrated["response"] = record["response"]
        migrated["hint"] = record["response"]  # hint = response for CoT data

    # Fields to skip
    skip_fields = {"choices", "task_id", "full_prompt", "question_with_choices"}

    # Keep all other fields
    for key, value in record.items():
        if key not in skip_fields and key not in migrated:
            migrated[key] = value

    return migrated


def migrate_file(input_path: Path, output_path: Path):
    """Migrate a single JSONL file."""
    print(f"Migrating {input_path.name}...")

    migrated_records = []
    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                migrated = migrate_record(record)
                migrated_records.append(migrated)
            except Exception as e:
                print(f"  Error on line {line_num}: {e}", file=sys.stderr)

    # Write migrated records
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in migrated_records:
            f.write(json.dumps(record) + "\n")

    print(f"  Wrote {len(migrated_records)} records to {output_path.name}")


def main():
    cot_dir = Path(__file__).parent.parent / "christine_experiments/data/cot"

    if not cot_dir.exists():
        print(f"Error: {cot_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Find all JSONL files
    jsonl_files = list(cot_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {cot_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(jsonl_files)} files to migrate\n")

    for input_path in jsonl_files:
        # Create backup
        backup_path = input_path.with_suffix(".jsonl.bak")
        if not backup_path.exists():
            backup_path.write_text(input_path.read_text())
            print(f"Created backup: {backup_path.name}")

        # Migrate in place
        migrate_file(input_path, input_path)
        print()

    print("Migration complete!")


if __name__ == "__main__":
    main()
