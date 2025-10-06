from utils.setup import setup_env
from inspect_ai import eval
from environments.math.math import math

setup_env()

# Configuration
LOG_DIR = "./math"
MODEL = "anthropic/claude-sonnet-4-5-20250929"
LIMIT = None
MAX_CONNECTIONS = 25
TEMPLATE = None  # Set to custom template string, or None for default
SPLIT = "train"  # Options: "test", "train", "validation"

# Example template (must include {prompt} placeholder):
# TEMPLATE = """Solve the following math problem. Put your final answer after "ANSWER: ".
#
# {prompt}
#
# Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER"."""

if __name__ == "__main__":
    log = eval(
        math(template=TEMPLATE, split=SPLIT),
        model=MODEL,
        log_dir=LOG_DIR,
        limit=LIMIT,
        max_connections=MAX_CONNECTIONS,
        display="rich",
    )
