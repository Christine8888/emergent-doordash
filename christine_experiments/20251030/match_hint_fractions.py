#!/usr/bin/env python3
"""
Match ID.json files to their closest hint fraction based on choice accuracy/stderr.
"""

import json
import re
from pathlib import Path
import numpy as np

def is_id_json(filename):
    """Check if filename is a 22-character ID JSON file."""
    pattern = r'^[A-Za-z0-9]{22}\.json$'
    return bool(re.match(pattern, filename))

def find_all_id_jsons(base_path):
    """Find all ID.json files recursively."""
    base = Path(base_path)
    id_jsons = []

    for json_file in base.rglob("*.json"):
        if is_id_json(json_file.name):
            id_jsons.append(json_file)

    return id_jsons

def find_hint_jsons(folder):
    """Find all gpqa_diamond_0shot_{hint}.json files in a folder."""
    hint_jsons = {}

    for json_file in folder.glob("gpqa_diamond_0shot_*.json"):
        # Extract hint fraction from filename
        match = re.search(r'gpqa_diamond_0shot_([0-9.]+)\.json$', json_file.name)
        if match:
            hint_fraction = float(match.group(1))
            hint_jsons[hint_fraction] = json_file

    return hint_jsons

def compute_mse(data1, data2):
    """Compute MSE between two choice results."""
    acc1 = data1["choice"]["accuracy"]
    stderr1 = data1["choice"]["stderr"]

    acc2 = data2["choice"]["accuracy"]
    stderr2 = data2["choice"]["stderr"]

    # Sum of squared errors
    mse = (acc1 - acc2)**2 + (stderr1 - stderr2)**2
    return mse

def find_best_match(id_data, hint_jsons):
    """Find the hint fraction with the closest match."""
    best_hint = None
    best_mse = float('inf')

    for hint_fraction, hint_path in hint_jsons.items():
        with open(hint_path, 'r') as f:
            hint_data = json.load(f)

        mse = compute_mse(id_data, hint_data)

        if mse < best_mse:
            best_mse = mse
            best_hint = hint_fraction

    return best_hint, best_mse

def main():
    base_path = "/Users/christineye/emergent-doordash/christine_experiments/20251015/results"

    # Find all ID.json files
    id_jsons = find_all_id_jsons(base_path)
    print(f"Found {len(id_jsons)} ID.json files")

    # Process each ID.json
    for id_json_path in id_jsons:
        print(f"\nProcessing: {id_json_path}")

        # Load the ID.json data
        with open(id_json_path, 'r') as f:
            id_data = json.load(f)

        # Check if it already has hint_fraction
        if "hint_fraction" in id_data:
            print(f"  Already has hint_fraction: {id_data['hint_fraction']}, skipping")
            continue

        # Find hint JSONs in the same folder
        folder = id_json_path.parent
        hint_jsons = find_hint_jsons(folder)

        if not hint_jsons:
            print(f"  No hint JSONs found in {folder}")
            continue

        print(f"  Found {len(hint_jsons)} hint JSONs: {sorted(hint_jsons.keys())}")

        # Find best match
        best_hint, best_mse = find_best_match(id_data, hint_jsons)

        print(f"  Best match: hint_fraction={best_hint}, MSE={best_mse:.6f}")
        print(f"  ID choice: acc={id_data['choice']['accuracy']:.4f}, stderr={id_data['choice']['stderr']:.4f}")

        # Add hint_fraction to the data
        id_data["hint_fraction"] = best_hint

        # Write back to file
        with open(id_json_path, 'w') as f:
            json.dump(id_data, f, indent=2)

        print(f"  ✓ Written hint_fraction={best_hint} to {id_json_path.name}")

if __name__ == "__main__":
    main()
