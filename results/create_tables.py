#!/usr/bin/env python
"""Utilities for aggregating eval.jsonl results into tabular form.

Usage examples:

  # Print a run-level summary as a Markdown table
  python results/create_tables.py markdown --table-type by_run

  # Use a different root directory (defaults to "results")
  python results/create_tables.py markdown --root other_results

The script is intentionally small but structured so you can easily
add new table types later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ruff: noqa: E402
from methods import method2class
from utils.logging import setup_logging

KNOWN_METHODS = set(method2class.keys())


def load_eval_results(root: Path) -> pd.DataFrame:
    """Load all *eval.jsonl files under ``root`` into a single DataFrame.

    The expected directory / filename layout is:

        root/<test_dataset>/<model>/<filename>

    with filenames of one of the forms

        <mas_method>_<eval_method>_eval.jsonl
        <mas_method>_<mas_config>_<eval_method>_eval.jsonl

    where ``mas_method`` is matched against known method names from
    ``methods.method2class`` and may contain underscores. Any remaining
    segments between ``mas_method`` and ``eval_method`` are treated as
    ``mas_config`` (which may also contain underscores or hyphens).

    Additional metadata columns are added to the resulting DataFrame:

    - ``test_dataset``
    - ``model``
    - ``mas_method``
    - ``mas_config`` (can be ``None`` if no explicit config segment)
    - ``eval_method``
    - ``eval_file`` (full path to the JSONL file)
    """

    root = root.resolve()
    all_dfs: list[pd.DataFrame] = []

    for path in root.rglob("*eval.jsonl"):
        # Expect: root/<test_dataset>/<model>/<filename>
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            # Path not under root; skip
            continue

        if len(rel_parts) < 3:
            # Unexpected layout; skip this file
            continue

        test_dataset = rel_parts[0]
        model = rel_parts[1]

        stem = path.stem  # e.g. "cot_xverify_eval" or "cot_config-mcp_xverify_eval"
        parts = stem.split("_")

        # Require filenames of the form:
        #   <mas_method>_<eval_method>_eval
        #   <mas_method>_<mas_config>_<eval_method>_eval
        # where <mas_method> is matched against known method names from
        # methods.method2class (and may itself contain underscores).
        if len(parts) < 3 or parts[-1] != "eval":
            # Not an eval file in the expected naming scheme
            continue

        eval_method = parts[-2]
        prefix_parts = parts[:-2]  # everything before eval_method + "eval"
        if not prefix_parts:
            # No method prefix at all
            logger.warning("Skipping eval file with no method prefix: {}", path)
            continue

        mas_method: Optional[str] = None
        mas_config: Optional[str] = None

        # Find the longest prefix that matches a known method name
        for i in range(len(prefix_parts), 0, -1):
            candidate = "_".join(prefix_parts[:i])
            if candidate in KNOWN_METHODS:
                mas_method = candidate
                remaining = prefix_parts[i:]
                if remaining:
                    mas_config = "_".join(remaining)
                break

        if mas_method is None:
            # No known method prefix in this filename; skip and log a warning.
            logger.warning(
                "Skipping eval file with unknown MAS method in filename '{}': {}",
                stem,
                path,
            )
            continue

        # Read this JSONL file
        df = pd.read_json(path, lines=True)

        # Extract flat token statistics, defaulting to 0 when missing.
        # Support both legacy "token_stats" and current "stats" keys.
        stats_columns = [
            "num_llm_calls",
            "prompt_tokens",
            "completion_tokens",
            "num_tool_calls",
        ]

        stats_field: Optional[str] = None
        if "token_stats" in df.columns:
            stats_field = "token_stats"
        elif "stats" in df.columns:
            stats_field = "stats"

        if stats_field is not None:

            def _extract_token_stats(cell):
                base = {k: 0 for k in stats_columns}
                if isinstance(cell, dict):
                    stats_dict = None
                    # Prefer stats for this model if present
                    if model in cell and isinstance(cell[model], dict):
                        stats_dict = cell[model]
                    elif len(cell) == 1:
                        # Fall back to the single entry if there is exactly one
                        only = next(iter(cell.values()))
                        if isinstance(only, dict):
                            stats_dict = only
                    if stats_dict is not None:
                        for key in base:
                            val = stats_dict.get(key)
                            if isinstance(val, (int, float)):
                                base[key] = int(val)
                return pd.Series(base)

            stats_df = df[stats_field].apply(_extract_token_stats)
        else:
            # No stats column at all; fill zeros
            stats_df = pd.DataFrame(0, index=df.index, columns=stats_columns)

        df = pd.concat([df, stats_df], axis=1)

        # Attach reconstructed metadata
        df["test_dataset"] = test_dataset
        df["model"] = model
        df["mas_method"] = mas_method
        df["mas_config"] = mas_config
        df["eval_method"] = eval_method
        df["eval_file"] = str(path)

        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError(f"No eval.jsonl files found under {root}")

    return pd.concat(all_dfs, ignore_index=True)


def build_by_run_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results at the (dataset, model, method, config, eval) level.

    The returned DataFrame has one row per unique combination of
    (test_dataset, model, mas_method, mas_config, eval_method), with
    simple accuracy statistics based on the ``eval_score`` field (if present).
    """

    group_cols = ["test_dataset", "model", "mas_method", "mas_config", "eval_method"]

    if "eval_score" not in df.columns:
        # Still return counts, but without accuracy metrics
        grouped = (
            df.groupby(group_cols, dropna=False).size().reset_index(name="n_examples")
        )
        return grouped

    # Ensure eval_score is numeric (e.g. 0/1)
    scores = pd.to_numeric(df["eval_score"], errors="coerce")
    df = df.copy()
    df["eval_score"] = scores

    grouped = df.groupby(group_cols, dropna=False)

    # Basic accuracy metrics
    acc = grouped["eval_score"].agg(n_examples="size", mean_score="mean")

    # Token statistics (use 0 for missing / non-numeric)
    for col in [
        "num_llm_calls",
        "prompt_tokens",
        "completion_tokens",
        "num_tool_calls",
    ]:
        if col in df.columns:
            acc[f"mean_{col}"] = grouped[col].mean()

    agg = acc.reset_index()
    agg = agg.rename(columns=lambda c: c[5:] if c.startswith("mean_") else c)

    return agg


def _format_for_markdown(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column-specific numeric formatting for Markdown output."""
    formatted = df.copy()

    # prompt_tokens / completion_tokens: integer (0 decimals)
    for col in ["prompt_tokens", "completion_tokens"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda x: "" if pd.isna(x) else f"{int(round(x))}"
            )

    # num_llm_calls / num_tool_calls: 1 decimal
    for col in ["num_llm_calls", "num_tool_calls"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.1f}"
            )

    # score: 3 decimals
    if "score" in formatted.columns:
        formatted["score"] = formatted["score"].map(
            lambda x: "" if pd.isna(x) else f"{x:.3f}"
        )

    return formatted


def format_markdown_by_dataset_model(
    table_df: pd.DataFrame, include_eval_method: bool = False
) -> str:
    """Render one Markdown table per (test_dataset, model) pair.

    The ``test_dataset`` and ``model`` columns are used only for grouping
    and are omitted from each table. The total number of examples ``N``
    for each (dataset, model) group is shown in the section heading.

    By default the ``eval_method`` and ``n_examples`` columns are omitted
    from the tables; ``eval_method`` can be kept via ``include_eval_method``.
    """

    group_cols = ["test_dataset", "model"]
    missing = [c for c in group_cols if c not in table_df.columns]
    if missing:
        raise ValueError(f"Expected columns missing from table_df: {missing}")

    lines: list[str] = []
    for (dataset, model), sub in table_df.groupby(group_cols, dropna=False):
        # Derive an N-label from n_examples if present
        n_label = ""
        if "n_examples" in sub.columns:
            n_vals = sub["n_examples"].dropna().astype(int)
            if not n_vals.empty:
                min_n = int(n_vals.min())
                max_n = int(n_vals.max())
                if min_n == max_n:
                    n_label = f" (N={min_n})"
                else:
                    n_label = f" (N={min_n}-{max_n})"

        # Drop grouping columns and optionally eval_method / n_examples
        drop_cols = list(group_cols)
        if "n_examples" in sub.columns:
            drop_cols.append("n_examples")
        if not include_eval_method and "eval_method" in sub.columns:
            drop_cols.append("eval_method")

        display = sub.drop(columns=drop_cols, errors="ignore")

        # Apply column-specific numeric formatting for Markdown
        display = _format_for_markdown(display)

        lines.append(f"### {dataset} / {model}{n_label}")
        lines.append("")
        lines.append(display.to_markdown(index=False))

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def command_markdown(args: argparse.Namespace) -> None:
    """Entry point for the ``markdown`` subcommand.

    Currently supports one table type:
    - "by_run": one row per (dataset, model, method, config, eval).

    Additional table types can be added later as needed.
    """

    root = Path(args.root)
    df = load_eval_results(root)

    if args.table_type == "by_run":
        table_df = build_by_run_table(df)
        markdown = format_markdown_by_dataset_model(
            table_df, include_eval_method=args.include_eval_method
        )
    else:
        raise SystemExit(f"Unknown table type: {args.table_type}")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate eval.jsonl results and optionally emit Markdown tables.",
    )

    parser.add_argument(
        "--root",
        type=str,
        default="results",
        help="Root directory containing eval.jsonl files (default: results)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Markdown table generator
    p_md = subparsers.add_parser(
        "markdown",
        help="Generate Markdown tables from aggregated results.",
    )
    p_md.add_argument(
        "--table-type",
        choices=["by_run"],
        default="by_run",
        help="Type of table to generate (default: by_run)",
    )
    p_md.add_argument(
        "--include-eval-method",
        action="store_true",
        help="Include eval_method as a column in Markdown tables (default: omit).",
    )
    p_md.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional path to write the Markdown table to; prints to stdout if omitted.",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging()

    if args.command == "markdown":
        command_markdown(args)
    else:
        parser.error(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
