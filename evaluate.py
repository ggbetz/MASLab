import argparse
import asyncio
import datetime
import json
import os
import threading

from loguru import logger
from tqdm import tqdm

from evaluations import get_eval_func
from inference import build_output_path
from methods import get_method_class
from utils import (
    load_model_api_config,
    read_valid_jsonl,
    redact_model_api_entry,
    reserve_unprocessed_queries,
    write_to_jsonl,
)
from utils.logging import setup_logging


async def evaluate_sample_async(args, item, save_eval_path, lock=None, llm=None):
    """Run a single evaluation using an async eval function and LLM."""
    eval_func = get_eval_func(args.eval_protocol, args.tested_dataset_name)

    if "response" in item:
        eval_content, eval_score = await eval_func(item, llm)
    else:
        eval_content, eval_score = "Infer Error", None

    save_data = item.copy()
    save_data["eval_content"] = eval_content
    save_data["eval_score"] = eval_score
    if args.debug:
        logger.info(json.dumps(save_data, indent=4))
    else:
        write_to_jsonl(save_eval_path, save_data, lock=lock)


async def main_async():
    parser = argparse.ArgumentParser()
    # args related to the evaluation
    parser.add_argument(
        "--eval_protocol",
        type=str,
        default="xverify",
        help="The evaluation protocol to be used.",
    )

    # args related to the model
    parser.add_argument(
        "--model_name",
        type=str,
        default="xverify-9b-c",
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

    # args related to evaluated objects
    parser.add_argument(
        "--tested_dataset_name",
        type=str,
        default="example_math",
        help="The dataset to be used for testing.",
    )
    parser.add_argument(
        "--tested_method_name", type=str, default="vanilla", help="MAS name."
    )
    parser.add_argument(
        "--tested_method_config_name",
        type=str,
        default=None,
        help="The config name for the method.",
    )
    parser.add_argument(
        "--tested_mas_model_name",
        type=str,
        default="llama-3.3-70b-instruct",
        help="The agent backend to be used for inference.",
    )
    parser.add_argument(
        "--tested_infer_path", type=str, default=None, help="Path to the output file."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Turn this on to run one defined sample for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Turn this on to overwrite the existing output file.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Turn this on to run the evaluation sequentially.",
    )
    args = parser.parse_args()
    time = datetime.datetime.now().isoformat()
    log_file = f"logs/inference_{args.test_dataset_name}_{args.method_name}_{args.model_name}_{time}.log"
    setup_logging(log_file=log_file)

    general_config = vars(args)

    logger.info(
        f"Evaluating {args.tested_method_name} on {args.tested_dataset_name} with {args.tested_mas_model_name} as MAS model using {args.model_name} as LLM"
    )
    logger.debug(f"General config: {json.dumps(general_config, indent=2)}")

    # Load model config
    model_api_config = load_model_api_config(args.model_api_config, args.model_name)
    general_config.update({"model_api_config": model_api_config})

    redacted_config = redact_model_api_entry(model_api_config[args.model_name])
    logger.info(f"Model API config: {redacted_config}")

    LLM_METHOD = get_method_class("vanilla")

    # Load evaluation data
    tested_infer_path = build_output_path(
        method_name=args.tested_method_name,
        method_config_name=args.tested_method_config_name,
        test_dataset_name=args.tested_dataset_name,
        model_name=args.tested_mas_model_name,
        output_path=args.tested_infer_path,
    )
    # Use eval_protocol to determine suffix, e.g. xverify_eval, foo_eval, etc.
    eval_suffix = f"{args.eval_protocol}_eval"
    save_eval_path = tested_infer_path.replace("infer", eval_suffix)

    if args.debug:
        sample = {"query": "1+3=?", "gt": "4", "response": "\\boxed{4}"}
        llm = LLM_METHOD(general_config)
        await evaluate_sample_async(args, sample, save_eval_path, lock=None, llm=llm)
    else:
        eval_data = read_valid_jsonl(tested_infer_path)
        logger.info(f"Before filtering: {len(eval_data)} samples")

        if args.overwrite and os.path.exists(save_eval_path):
            os.remove(save_eval_path)
            logger.warning(f"Removing existing evaluation file: {save_eval_path}")
        else:
            eval_data = reserve_unprocessed_queries(save_eval_path, eval_data)
        logger.info(f"After filtering: {len(eval_data)} samples to evaluate")

        lock = threading.Lock()

        llm = LLM_METHOD(general_config)
        if args.sequential:
            for sample in eval_data:
                await evaluate_sample_async(args, sample, save_eval_path, lock, llm)
        else:
            tasks = [
                evaluate_sample_async(args, sample, save_eval_path, lock, llm)
                for sample in eval_data
            ]

            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Evaluating MAS",
            ):
                await coro

        # Load evaluation results and print the statistics
        with open(save_eval_path, "r") as f:
            saved_data = [json.loads(line) for line in f.readlines()]

        sample_num = len(saved_data)
        valid_eval_score_list = [
            sample["eval_score"]
            for sample in saved_data
            if sample["eval_score"] is not None
        ]
        valid_correct_num = sum([1 for score in valid_eval_score_list if score == 1])
        num_valid = len(valid_eval_score_list)
        num_exclude_eval_error = len(
            [
                sample
                for sample in saved_data
                if not sample["eval_content"].startswith("Eval Error")
            ]
        )
        logger.info(
            f"Evaluation completed - Total: {sample_num} | Valid: {num_valid} | Correct: {valid_correct_num} | Accuracy: {valid_correct_num / num_valid * 100:.2f}%"
        )
        logger.info(
            f"Excluding eval errors - Valid: {num_exclude_eval_error} | Accuracy: {valid_correct_num / num_exclude_eval_error * 100:.2f}%"
        )


if __name__ == "__main__":
    asyncio.run(main_async())
