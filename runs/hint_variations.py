"""
I want to try making different variations of hints for the AIME 2025 dataset at different hint levels.

Hint ideas:
1. truncation
2. masking (word level, sentence level)
3. bag of hints
4. masking --> clean statement
"""
from src.datasets import AIME2025
from src.inference import query_model_anthropic_batch
from src.results import ResultEntry, ResultsLog
import re

def extract_answer(text: str):
    """
    Extracts the answer found between <answer> and </answer> tags in the given text.
    Returns the extracted answer as a string, or None if not found.
    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def grade_answer(model_answer: str, answer: str) -> bool:
    model_answer = model_answer.strip()

    # strip \boxed{...}
    match = re.search(r"\\boxed\{(.+?)\}", model_answer)
    if match:
        model_answer = match.group(1)

    return model_answer == answer.strip()


def main():
    dataset = AIME2025.load()

    # use 2 questions to test on
    num_q_to_test = 1
    aime_results = ResultsLog()
    for i in range(num_q_to_test):
        row = dataset[i]

        prompts = [
            row.question + "Put your final answer inbetween <answer></answer> tags"
        ]

        messages = query_model_anthropic_batch(
            model="claude-opus-4-6",
            prompts=prompts,
            max_tokens=32000,
        )
        for msg in messages:
            text_response = msg.response_text
            model_answer = extract_answer(text_response)
            if grade_answer(model_answer, row['answer']):
                print(f"Question {i} passed")
            else:
                print(f"Question {i} failed")

            aime_results.add(ResultEntry(
                datum_id=row.id,
                question=row.question,
                ground_truth_answer=row.answer,
                prompt=prompts[0],
                model="claude-opus-4-6",
                response_text=text_response,
                extracted_answer=model_answer,
                is_correct=grade_answer(model_answer, row.answer),
                input_token_count=msg.input_token_count,
                output_token_count=msg.output_token_count,
                cost=msg.cost,
                is_error=msg.is_error,
            ))
    
    aime_results.save("aime_results.json")
        

if __name__ == "__main__":
    # python -m runs.hint_variations
    main()
