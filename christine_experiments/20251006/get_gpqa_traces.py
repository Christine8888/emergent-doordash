from utils.setup import setup_env
from inspect_ai import eval
from environments.gpqa.gpqa import gpqa_diamond

setup_env()


# Configuration
LOG_DIR = "./gpqa"
MODEL = "anthropic/claude-sonnet-4-5-20250929"
LIMIT = None
MAX_CONNECTIONS = 15
TEMPLATE = None  # Set to custom template string, or None for default

if __name__ == "__main__":
    log = eval(
        gpqa_diamond(template=TEMPLATE),
        model=MODEL,
        log_dir=LOG_DIR,
        limit=LIMIT,
        max_connections=MAX_CONNECTIONS,
        display="rich",
    )
