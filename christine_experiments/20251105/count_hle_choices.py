#!/usr/bin/env python3
"""Count answer choices in HLE questions and compute average guessing accuracy."""

import json
import re
from pathlib import Path

def count_answer_choices(question_text: str) -> int:
    """Count unique answer choices in question text.

    Args:
        question_text: The question text containing "Answer Choices:"

    Returns:
        Number of unique answer choices (A, B, C, etc.)
    """
    # Find the "Answer Choices:" section
    match = re.search(r'Answer Choices:', question_text, re.IGNORECASE)
    if not match:
        return 0

    # Get text after "Answer Choices:"
    choices_section = question_text[match.end():]

    # Find all capital letters that appear at the start of a line or after newline
    # Pattern: newline (or start) followed by capital letter followed by period or dot
    choice_pattern = r'^([A-Z])\.|\n([A-Z])\.'
    matches = re.findall(choice_pattern, choices_section, re.MULTILINE)

    # Flatten the tuple matches and get unique letters
    letters = set()
    for match_tuple in matches:
        for letter in match_tuple:
            if letter:  # Skip empty strings from non-matching groups
                letters.add(letter)

    return len(letters)

def main():
    data_path = Path("/Users/christineye/emergent-doordash/christine_experiments/data/cot/hle_filtered.jsonl")

    if not data_path.exists():
        print(f"Error: File not found: {data_path}")
        return

    total_questions = 0
    choice_counts = []

    with open(data_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                question = data.get('question', '')

                num_choices = count_answer_choices(question)
                if num_choices > 0:
                    choice_counts.append(num_choices)
                    total_questions += 1
                else:
                    print(f"Warning: No choices found in line {line_num}")

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")

    if not choice_counts:
        print("No questions with answer choices found!")
        return

    # Calculate statistics
    avg_choices = sum(choice_counts) / len(choice_counts)
    avg_guessing_accuracy = sum(1/n for n in choice_counts) / len(choice_counts)

    # Count distribution
    from collections import Counter
    distribution = Counter(choice_counts)

    print(f"\nTotal questions: {total_questions}")
    print(f"\nAnswer choice distribution:")
    for num_choices in sorted(distribution.keys()):
        count = distribution[num_choices]
        print(f"  {num_choices} choices: {count} questions ({count/total_questions*100:.1f}%)")

    print(f"\nAverage number of choices: {avg_choices:.2f}")
    print(f"Average guessing accuracy: {avg_guessing_accuracy:.4f} ({avg_guessing_accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()
