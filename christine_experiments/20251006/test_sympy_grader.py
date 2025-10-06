import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from environments.math.utils import is_equiv_sympy, normalize_final_answer

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

async def test_cases():
    test_pairs = [
        ("75", "75"),
        ("75", "75.0"),
        ("1/2", "0.5"),
        ("2x+3", "3+2x"),
        ("x^2+2x+1", "(x+1)^2"),
        ("\\frac{1}{2}", "0.5"),
        ("100", "100"),
        ("3.14", "3.14"),
    ]

    for answer, target in test_pairs:
        print(f"\n{'='*60}")
        print(f"Testing: answer='{answer}', target='{target}'")
        print(f"{'='*60}")

        # Test normalization
        norm_answer = await normalize_final_answer(answer)
        norm_target = await normalize_final_answer(target)
        print(f"Normalized answer: '{norm_answer}'")
        print(f"Normalized target: '{norm_target}'")

        # Test sympy equivalence
        result = await is_equiv_sympy(norm_answer, norm_target)
        print(f"Result: {result}")
        print()

if __name__ == "__main__":
    asyncio.run(test_cases())
