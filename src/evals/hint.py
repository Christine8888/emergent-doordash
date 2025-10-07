def get_prefill_fraction(reasoning: str, fraction: float = 0.5) -> str:
    # naive approach: split by words --> get fraction of words
    words = reasoning.split()
    return " ".join(words[: int(len(words) * fraction)])