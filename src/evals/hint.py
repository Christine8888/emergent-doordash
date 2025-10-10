import re

def get_prefill_fraction(reasoning: str, fraction: float = 0.5, stop_string: str = "ANSWER:") -> str:
    # Split on whitespace while capturing it
    tokens = re.split(r'(\s+)', reasoning)
    
    # Filter to just words (non-whitespace tokens)
    words = [t for t in tokens if t and not t.isspace()]
    num_words = int(len(words) * fraction)
    
    # Reconstruct up to the target word count
    result = []
    word_count = 0
    for token in tokens:
        if not token.isspace():
            if word_count >= num_words or token == stop_string:
                break
            word_count += 1
        result.append(token)
    
    return "".join(result)