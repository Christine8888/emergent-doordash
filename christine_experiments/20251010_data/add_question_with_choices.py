#!/usr/bin/env python3
"""
Add question_with_choices field to JSONL entries that are missing it.
Extracts from full_prompt by removing instruction prefix and solution suffix.
"""

import json
import sys
from pathlib import Path

# The prefix to remove from full_prompt
PREFIX = """Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of the options. Think step by step before answering.

PROBLEM:
"""

# The suffix to remove from full_prompt
SUFFIX = "\n\nSOLUTION:"


def extract_question_with_choices(full_prompt: str) -> str:
    """Extract question_with_choices from full_prompt."""
    result = full_prompt

    # Remove prefix if present
    if result.startswith(PREFIX):
        result = result[len(PREFIX):]

    # Remove suffix if present
    if result.endswith(SUFFIX):
        result = result[:-len(SUFFIX)]

    return result


def process_file(input_file: Path, output_file: Path):
    """Process JSONL file and add missing question_with_choices fields."""
    added_count = 0
    total_count = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            total_count += 1
            data = json.loads(line)

            # Add question_with_choices if missing
            if 'question_with_choices' not in data and 'full_prompt' in data:
                data['question_with_choices'] = extract_question_with_choices(data['full_prompt'])
                added_count += 1

            # Write back to output
            outfile.write(json.dumps(data) + '\n')

    print(f"Processed {total_count} entries")
    print(f"Added question_with_choices to {added_count} entries")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_question_with_choices.py <input_file> [output_file]")
        print("If output_file is not provided, will overwrite input file")
        sys.exit(1)

    input_file = Path(sys.argv[1])

    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        # Create temporary file, then rename
        output_file = input_file.with_suffix('.tmp')

    process_file(input_file, output_file)

    # If overwriting, rename temp file to original
    if len(sys.argv) < 3:
        output_file.replace(input_file)
        print(f"Updated {input_file}")
    else:
        print(f"Output written to {output_file}")
