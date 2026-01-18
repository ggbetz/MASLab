import traceback

import yaml
from loguru import logger


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def handle_retry_error(retry_state):
    err_msg = "Retry failed"
    if retry_state.outcome:
        exc = retry_state.outcome.exception()
        err_msg += (
            f"Final exception after retries: {repr(exc)}. "
            f"Traceback: {traceback.format_exception(type(exc), exc, exc.__traceback__)}"
        )
    logger.error(err_msg)
    return None
