import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_level="INFO", log_file=None):
    logger.remove()  # Remove default handler
    logger.add(sys.stdout, level=log_level)
    if log_file:
        log_path = Path(log_file)
        # Create parent directories if they don't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, rotation="10 MB")
    return logger
