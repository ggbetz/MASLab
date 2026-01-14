import argparse
import asyncio
import json
import os
import threading
import traceback

from tqdm import tqdm

from methods import get_method_class
from utils import load_model_api_config, reserve_unprocessed_queries, write_to_jsonl


async def run_sample(mas, sample, output_path, lock):
    """Run a single sample using an async MAS instance."""
    save_data = sample.copy()
    try:
        mas_output = await mas.inference(sample)
        if "response" not in mas_output:
            raise ValueError(
                f"The key 'response' is not found in the MAS output: {mas_output}"
            )
        save_data.update(mas_output)
    except Exception:
        save_data["error"] = f"Inference Error: {traceback.format_exc()}"
    save_data.update({"token_stats": mas.get_token_stats()})
    write_to_jsonl(lock, output_path, save_data)


async def main_async():
    parser = argparse.ArgumentParser()
    # args related to the method
    parser.add_argument("--method_name", type=str, default="vanilla", help="MAS name.")
    parser.add_argument(
        "--method_config_name",
        type=str,
        default=None,
        help="The config file name. If None, the default config file will be used.",
    )

    # args related to the model
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="The agent backend to be used for inference.",
    )
    parser.add_argument(
        "--model_api_config",
        type=str,
        default="model_api_configs/model_api_config.json",
    )
    parser.add_argument(
        "--model_temperature", type=float, default=0.5, help="Temperature for sampling."
    )
    parser.add_argument(
        "--model_max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens for sampling.",
    )
    parser.add_argument(
        "--model_timeout", type=int, default=600, help="Timeout for sampling."
    )

    # args related to dataset
    parser.add_argument(
        "--test_dataset_name",
        type=str,
        default="example_math",
        help="The dataset to be used for testing.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="Path to the output file."
    )
    parser.add_argument("--require_val", action="store_true")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    general_config = vars(args)

    # Load model config
    model_api_config = load_model_api_config(args.model_api_config, args.model_name)
    general_config.update({"model_api_config": model_api_config})
    print("-" * 50, f"\n>> Model API config: {model_api_config[args.model_name]}")

    if args.debug:
        print(args.method_name)
        if args.method_name in ["dylan_humaneval"]:
            sample = {
                "query": """\
def add(a: int, b: int) -> int:
    \"\"\"Return the sum of a and b.\"\"\"
    # Write your code here
"""
            }
        else:
            sample = {
                "query": "If $|x+5|-|3x-6|=0$, find the largest possible value of $x$. Express your answer as an improper fraction."
            }
        MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
        async with MAS_METHOD(
            general_config, method_config_name=args.method_config_name
        ) as mas:
            response = await mas.inference(sample)
            print(json.dumps(response, indent=4))
            print(f"\n>> Token stats: {json.dumps(mas.get_token_stats(), indent=4)}")
    else:
        print(f">> Method: {args.method_name} | Dataset: {args.test_dataset_name}")

        # load dataset
        with open(f"./datasets/data/{args.test_dataset_name}.json", "r") as f:
            test_dataset = json.load(f)

        val_dataset = None
        if args.require_val:
            val_dataset_path = f"./datasets/data/{args.test_dataset_name}_val.json"
            if not os.path.exists(val_dataset_path):
                raise FileNotFoundError(
                    f"Validation dataset not found at {val_dataset_path}. Please provide a valid path."
                )
            with open(val_dataset_path, "r") as f:
                val_dataset = json.load(f)

        # get output path
        file_name = (
            f"{args.method_name}_{args.method_config_name}_infer.jsonl"
            if args.method_config_name
            else f"{args.method_name}_infer.jsonl"
        )
        output_path = (
            args.output_path
            if args.output_path is not None
            else f"./results/{args.test_dataset_name}/{args.model_name}/{file_name}"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # reserve unprocessed samples
        test_dataset = reserve_unprocessed_queries(output_path, test_dataset)
        print(f">> After filtering: {len(test_dataset)} samples")

        MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
        lock = threading.Lock()

        async with MAS_METHOD(
            general_config, method_config_name=args.method_config_name
        ) as mas:
            # Optional optimization step
            if args.require_val and val_dataset is not None:
                await asyncio.to_thread(mas.optimizing, val_dataset)

            if args.sequential:
                for sample in test_dataset:
                    await run_sample(mas, sample, output_path, lock)
            else:
                tasks = [
                    run_sample(mas, sample, output_path, lock)
                    for sample in test_dataset
                ]
                for coro in tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc="Processing queries",
                ):
                    await coro


if __name__ == "__main__":
    asyncio.run(main_async())
