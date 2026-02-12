"""Configuration for MATH sampling script."""

from inspect_ai.dataset import Sample
from environments.math.math import get_math_dataset, DEFAULT_INSTRUCTIONS, DEFAULT_EXAMPLE_TEMPLATE
from environments.math.utils import extract_answer as math_extract_answer, grade_math_answer


def get_dataset(split: str = "train", shuffle: bool = False):
    """Load MATH dataset from local files."""
    return get_math_dataset(split=split, shuffle=shuffle)


def extract_answer(response: str) -> str:
    """Extract answer from response."""
    return math_extract_answer(response)


async def grade_answer(extracted_answer: str, target: str) -> bool:
    """Grade MATH answer using canonical math grader with sympy."""
    return await grade_math_answer(
        answer=extracted_answer,
        target=target,
        exact_match=True,
        use_sympy=True,
    )


def format_prompt(sample: Sample) -> str:
    """Format MATH prompt using solver template."""
    from hints.sample_utils import sample_input_to_str

    current_task = DEFAULT_EXAMPLE_TEMPLATE.format(
        question=sample_input_to_str(sample.input),
        solution="",
    )
    return DEFAULT_INSTRUCTIONS + "\n\n" + current_task


def extract_sample_fields(sample: Sample) -> dict:
    """Extract additional fields for sample_to_dict."""
    return {}


def add_cli_args(parser):
    """Add MATH-specific CLI arguments."""
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (train/test/validation)",
    )


def get_dataset_kwargs(args):
    """Get dataset kwargs from CLI args."""
    return {
        "split": args.split,
        "shuffle": False,
    }
