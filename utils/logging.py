import sys

from loguru import logger


def setup_logging(log_level="INFO", log_file=None):
    logger.remove()  # Remove default handler
    logger.add(sys.stdout, level=log_level)
    if log_file:
        logger.add(log_file, rotation="10 MB")
    return logger
