"inferency.py"

import argparse
import asyncio
import json
import os
import traceback
import uuid
from asyncio import Lock, Semaphore
from typing import Any, Dict, Iterable, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from methods import get_method_class
from utils import (
    load_model_api_config,
    redact_model_api_entry,
    reserve_unprocessed_queries,
    write_to_jsonl,
)
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
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

    # args related to agent configuration strictness
    parser.add_argument(
        "--agent_config_mode",
        type=str,
        choices=["strict", "warn"],
        default="warn",
        help=(
            "How to handle missing logical_agents entries for agent IDs. "
            "'strict' raises an error; 'warn' logs a warning and falls back "
            "to default agent settings."
        ),
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument(
        "--max_concurrent_samples",
        type=int,
        default=10,
        help="Maximum number of samples processed concurrently (ignored if --sequential is set).",
    )
    return parser.parse_args()


def build_general_config(args: argparse.Namespace) -> Dict[str, Any]:
    general_config: Dict[str, Any] = vars(args).copy()
    model_api_config = load_model_api_config(args.model_api_config, args.model_name)
    general_config["model_api_config"] = model_api_config

    redacted_config = redact_model_api_entry(model_api_config[args.model_name])
    logger.info("-" * 50, f"\n>> Model API config: {redacted_config}")
    return general_config


def load_datasets(
    args: argparse.Namespace,
) -> Tuple[Iterable[Dict[str, Any]], Optional[Iterable[Dict[str, Any]]]]:
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

    return test_dataset, val_dataset


def build_output_path(
    method_name: str,
    method_config_name: Optional[str],
    test_dataset_name: str,
    model_name: str,
    output_path: Optional[str],
) -> str:
    if method_config_name is not None:
        method_config_name = method_config_name.replace("_", "-")
    file_name = (
        f"{method_name}_{method_config_name}_infer.jsonl"
        if method_config_name
        else f"{method_name}_infer.jsonl"
    )
    if output_path is not None:
        return output_path
    return f"./results/{test_dataset_name}/{model_name}/{file_name}"


def create_mas(args: argparse.Namespace, general_config: Dict[str, Any]):
    mas_method = get_method_class(args.method_name, args.test_dataset_name)
    return mas_method(general_config, method_config_name=args.method_config_name)


async def run_sample(
    mas, sample: Dict[str, Any], output_path: str, write_lock: Lock
) -> None:
    """Run a single sample using an async MAS instance and persist results."""

    save_data = sample.copy()
    sample_uid = uuid.uuid4().hex

    try:
        async with mas.sample_context(sample_uid):
            mas_output = await mas.inference(sample)
        if "response" not in mas_output:
            raise ValueError(
                f"The key 'response' is not found in the MAS output: {mas_output}"
            )
        save_data.update(mas_output)
    except Exception as e:
        logger.error(f"Inference error for sample {sample_uid}: {e}")
        save_data["error"] = f"Inference Error: {traceback.format_exc()}"

    save_data["token_stats"] = mas.get_token_stats()

    async with write_lock:
        write_to_jsonl(output_path, save_data)


async def run_debug(mas, args: argparse.Namespace) -> None:
    logger.info(args.method_name)
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

    sample_uid = uuid.uuid4().hex
    async with mas.sample_context(sample_uid):
        response = await mas.inference(sample)

    logger.info(json.dumps(response, indent=4))
    logger.info(f"\n>> Token stats: {json.dumps(mas.get_token_stats(), indent=4)}")


async def run_full_inference(
    mas,
    args: argparse.Namespace,
    general_config: Dict[str, Any],
) -> None:
    logger.info(f">> Method: {args.method_name} | Dataset: {args.test_dataset_name}")

    test_dataset, val_dataset = load_datasets(args)

    output_path = build_output_path(
        method_name=args.method_name,
        method_config_name=args.method_config_name,
        test_dataset_name=args.test_dataset_name,
        model_name=args.model_name,
        output_path=args.output_path,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    test_dataset = reserve_unprocessed_queries(output_path, test_dataset)
    logger.info(f">> After filtering: {len(test_dataset)} samples")

    if args.require_val and val_dataset is not None:
        await asyncio.to_thread(mas.optimizing, val_dataset)

    write_lock = Lock()

    if args.sequential:
        for sample in test_dataset:
            await run_sample(mas, sample, output_path, write_lock)
    else:
        semaphore = Semaphore(args.max_concurrent_samples)

        async def bounded_run(sample):
            async with semaphore:
                await run_sample(mas, sample, output_path, write_lock)

        tasks = [bounded_run(sample) for sample in test_dataset]
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Processing queries",
        ):
            await coro


async def main_async() -> None:
    args = parse_args()
    setup_logging()
    general_config = build_general_config(args)
    mas = create_mas(args, general_config)

    if args.debug:
        await run_debug(mas, args)
    else:
        await run_full_inference(mas, args, general_config)


if __name__ == "__main__":
    asyncio.run(main_async())
