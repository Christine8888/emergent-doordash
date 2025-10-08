"""
Script to load MATH dataset from HuggingFace and save as JSONL with id field.

This exports the entire MATH dataset from HuggingFace in the same format,
just adding an "id" field (train_0, train_1, ..., test_0, test_1, etc.)
so the rest of the code can stay the same.
"""
from datasets import load_dataset
import json
from pathlib import Path

def export_math_dataset():
    """Load MATH dataset and export as JSONL files with id field."""

    # Load dataset from HuggingFace
    print("Loading MATH dataset from HuggingFace...")
    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")

    # Create output directory
    output_dir = Path("/nlp/scr/cye/emergent-doordash/src/environments/math/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process train and test splits
    for split_name in ["train", "test"]:
        split_data = dataset[split_name]
        output_file = output_dir / f"math_{split_name}.jsonl"

        print(f"\nProcessing {split_name} split: {len(split_data)} examples")
        print(f"Sample fields: {list(split_data[0].keys())}")

        with open(output_file, "w") as f:
            for idx, example in enumerate(split_data):
                # Create record with all original fields plus id
                record = dict(example)
                record["id"] = f"{split_name}_{idx}"

                # Write as JSONL
                f.write(json.dumps(record) + "\n")

        print(f"Saved to {output_file}")

if __name__ == "__main__":
    export_math_dataset()
