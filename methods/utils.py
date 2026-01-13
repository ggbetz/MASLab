import traceback

import yaml


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def handle_retry_error(retry_state):
    print("Retry failed")
    if retry_state.outcome:
        exc = retry_state.outcome.exception()
        print(f"Final exception after retries: {repr(exc)}")
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    return None
