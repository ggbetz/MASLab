from .evaluate_xverify import eval_func_xverify_async


def get_eval_func(eval_protocol, tested_dataset_name):
    if eval_protocol == "xverify":
        return eval_func_xverify_async
    else:
        raise ValueError(f"Unsupported evaluation protocol: {eval_protocol}")
