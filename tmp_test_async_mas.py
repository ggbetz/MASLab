import asyncio
import json

from methods import get_method_class
from utils import load_model_api_config


async def main():
    method_name = "cot"
    method_config_name = "config_mcp"
    test_dataset_name = "example_math"
    model_name = "kit.gpt-oss-120b"

    model_api_config = load_model_api_config(
        ".secrets/model_api_config.json", model_name
    )

    general_config = {
        "method_name": method_name,
        "method_config_name": method_config_name,
        "model_name": model_name,
        "model_api_config": model_api_config,
        "model_temperature": 0.5,
        "model_max_tokens": 2048,
        "model_timeout": 600,
        "test_dataset_name": test_dataset_name,
        "output_path": None,
        "require_val": False,
        "debug": True,
        "sequential": True,
    }

    MAS_METHOD = get_method_class(method_name, test_dataset_name)

    async with MAS_METHOD(general_config, method_config_name=method_config_name) as mas:
        sample = {
            "query": "If $|x+5|-|3x-6|=0$, find the largest possible value of $x$. Express your answer as an improper fraction."
        }
        out = mas.inference(sample)
        print(json.dumps(out, indent=4))
        print(json.dumps(mas.get_token_stats(), indent=4))


if __name__ == "__main__":
    asyncio.run(main())
