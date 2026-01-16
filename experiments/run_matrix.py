"""Orchestrate MASLab experiments defined in a YAML matrix.

Typical usage (run from repo root):

- Dry run (show what would be executed):

    uv run python experiments/run_matrix.py --dry-run

- Use an alternate config in the same dir

    uv run python experiments/run_matrix.py --config experiments/my_experiments.yaml

- Run full pipeline (build datasets, then inference, then evaluation):

    uv run python experiments/run_matrix.py

- Only build datasets (no inference or evaluation):

    uv run python experiments/run_matrix.py --only build

- Only run inference (build if needed, skip evaluation):

    uv run python experiments/run_matrix.py --only infer

- Only run evaluation (build + inference if needed):

    uv run python experiments/run_matrix.py --only eval

The experiments, datasets, methods, and models are configured in
`experiments/experiments.yaml`.
"""

import argparse
import itertools
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ruff: noqa: E402
from mas_datasets_config import dataset_is_supported
from utils.logging import setup_logging


@dataclass
class BuildConfig:
    num2sample: Optional[int] = None


@dataclass
class InferenceConfig:
    require_val: bool = False
    sequential: bool = False
    agent_config_mode: str = "warn"


@dataclass
class EvalConfig:
    overwrite: bool = False
    sequential: bool = False


@dataclass
class DatasetMethod:
    method: str
    config: Optional[str] = None
    inference: Optional[Dict[str, Any]] = None


@dataclass
class DatasetConfig:
    name: str
    methods: List[DatasetMethod]


@dataclass
class MatrixConfig:
    api_config: str
    eval_model: str
    eval_protocol: str
    defaults: Dict[str, Any]
    datasets: List[DatasetConfig]
    inference_models: List[str]


Status = Literal["pending", "running", "done", "skipped", "failed"]


@dataclass
class InferJob:
    dataset: str
    method: str
    config_name: Optional[str]
    mas_model: str
    infer_path: Path
    agent_config_mode: str = "warn"
    status: Status = "pending"
    proc: Optional[subprocess.Popen] = None


@dataclass
class EvalJob:
    dataset: str
    method: str
    config_name: Optional[str]
    mas_model: str
    infer_path: Path
    eval_path: Path
    status: Status = "pending"
    proc: Optional[subprocess.Popen] = None


def _parse_dataset_method(m: Dict[str, Any]) -> DatasetMethod:
    return DatasetMethod(
        method=m["method"],
        config=m.get("config"),
        inference=m.get("inference"),
    )


def validate_matrix(matrix: MatrixConfig) -> None:
    errors: list[str] = []

    # dataset existence check
    for ds in matrix.datasets:
        if not dataset_is_supported(ds.name):
            errors.append(
                f"Dataset '{ds.name}' not supported by datasets/build_test_dataset.py"
            )

    if errors:
        for error in errors:
            logger.error(error)
        raise ValueError("Matrix validation failed")


def load_yaml_config(path: Path) -> MatrixConfig:
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    api_config = raw["api_config"]
    eval_model = raw.get("eval_model", "xverify-9b-c")
    eval_protocol = raw.get("eval_protocol", "xverify")
    defaults = raw.get("defaults", {})

    # Global method groups: map alias -> list of DatasetMethod
    raw_method_groups = raw.get("method_groups") or {}
    method_groups: Dict[str, List[DatasetMethod]] = {}
    for alias, entries in raw_method_groups.items():
        if entries is None:
            entries = []
        if not isinstance(entries, list):
            raise TypeError(f"method_groups[{alias!r}] must be a list of methods")
        parsed_entries: List[DatasetMethod] = []
        for m in entries:
            parsed_entries.append(_parse_dataset_method(m))
        method_groups[alias] = parsed_entries

    datasets: List[DatasetConfig] = []
    for ds in raw.get("datasets", []):
        methods: List[DatasetMethod] = []

        # 1) Expand global method groups referenced by this dataset
        for alias in ds.get("method_groups", []) or []:
            if alias not in method_groups:
                raise ValueError(f"Unknown method group alias: {alias}")
            methods.extend(method_groups[alias])

        # 2) Add dataset-specific methods
        for m in ds.get("methods", []) or []:
            methods.append(_parse_dataset_method(m))

        datasets.append(DatasetConfig(name=ds["name"], methods=methods))

    inference_models = list(raw.get("inference_models", []))

    return MatrixConfig(
        api_config=api_config,
        eval_model=eval_model,
        eval_protocol=eval_protocol,
        defaults=defaults,
        datasets=datasets,
        inference_models=inference_models,
    )


def build_dataset(dataset: str, build_cfg: BuildConfig, dry_run: bool = False) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        "datasets/build_test_dataset.py",
        "--dataset_name",
        dataset,
    ]
    if build_cfg.num2sample is not None:
        cmd += ["--num2sample", str(build_cfg.num2sample)]

    logger.info("[build] {}", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def build_infer_cmd(
    dataset: str,
    method: str,
    config_name: Optional[str],
    mas_model: str,
    api_config: str,
    infer_cfg: InferenceConfig,
    agent_config_mode: str,
) -> List[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "inference.py",
        "--test_dataset_name",
        dataset,
        "--method_name",
        method,
        "--model_api_config",
        api_config,
        "--model_name",
        mas_model,
    ]
    if config_name is not None:
        cmd += ["--method_config_name", config_name]

    # Pass agent configuration strictness through to inference.py
    if agent_config_mode:
        cmd += ["--agent_config_mode", agent_config_mode]

    if infer_cfg.require_val:
        cmd.append("--require_val")
    if infer_cfg.sequential:
        cmd.append("--sequential")
    return cmd


def build_eval_cmd(
    dataset: str,
    method: str,
    config_name: Optional[str],
    mas_model: str,
    api_config: str,
    eval_model: str,
    eval_protocol: str,
    eval_cfg: EvalConfig,
) -> List[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "evaluate.py",
        "--eval_protocol",
        eval_protocol,
        "--tested_dataset_name",
        dataset,
        "--tested_method_name",
        method,
        "--tested_mas_model_name",
        mas_model,
        "--model_name",
        eval_model,
        "--model_api_config",
        api_config,
    ]
    if config_name is not None:
        cmd += ["--tested_method_config_name", config_name]
    if eval_cfg.overwrite:
        cmd.append("--overwrite")
    if eval_cfg.sequential:
        cmd.append("--sequential")
    return cmd


def build_output_path(
    dataset: str,
    method: str,
    config_name: Optional[str],
    mas_model: str,
) -> Path:
    # Mirror inference.build_output_path: results/{dataset}/{model}/{method}[_config]_infer.jsonl
    if config_name is not None:
        normalized_config = config_name.replace("_", "-")
        file_name = f"{method}_{normalized_config}_infer.jsonl"
    else:
        file_name = f"{method}_infer.jsonl"
    return REPO_ROOT / "results" / dataset / mas_model / file_name


def build_infer_done_path(infer_path: Path) -> Path:
    """Return the sidecar ".done" marker path for an inference output file."""
    return infer_path.with_suffix(infer_path.suffix + ".done")


def infer_output_complete(infer_path: Path) -> bool:
    """Return True if the inference output has a corresponding .done marker.

    This relies on inference.py's behavior of creating `<output>.done` only after
    successfully processing all samples. For legacy runs that predate the .done
    convention, callers may still want to fall back to `infer_path.exists()`.
    """
    done_path = build_infer_done_path(infer_path)
    return infer_path.exists() and done_path.exists()


def parse_bool_flag(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.lower() in {"1", "true", "yes", "y"}
    return bool(raw)


def run_scheduler(
    infer_jobs: List[InferJob],
    eval_jobs: List[EvalJob],
    matrix: MatrixConfig,
    infer_cfg: InferenceConfig,
    eval_cfg: EvalConfig,
    args: argparse.Namespace,
) -> int:
    active_infer: Optional[InferJob] = None
    active_eval: Optional[EvalJob] = None

    infer_index: Dict[tuple[str, str, Optional[str], str], InferJob] = {
        (j.dataset, j.method, j.config_name, j.mas_model): j for j in infer_jobs
    }

    def all_done(jobs: List[InferJob] | List[EvalJob]) -> bool:
        return all(j.status in ("done", "skipped", "failed") for j in jobs)

    def eval_ready(job: EvalJob) -> bool:
        if args.dry_run:
            return True

        key = (job.dataset, job.method, job.config_name, job.mas_model)
        infer_job = infer_index.get(key)

        if infer_job is not None:
            # Prefer file-based completion (using .done marker) over in-memory status.
            if infer_output_complete(infer_job.infer_path):
                return True
            if infer_job.status in ("failed", "skipped"):
                if job.status == "pending":
                    job.status = "skipped"
                    logger.warning(
                        "[eval] skip: infer job status is {} for {}",
                        infer_job.status,
                        infer_job.infer_path,
                    )
                return False
            # Inference job is still pending or running and has not produced a
            # complete output yet.
            return False

        # No infer job scheduled in this run; fall back to file-based completion
        # for legacy runs, where only the raw infer file may exist.
        if infer_output_complete(job.infer_path):
            return True
        return job.infer_path.exists()

    if not infer_jobs and not eval_jobs:
        logger.info("No inference or evaluation jobs to run.")
        return 0

    try:
        while True:
            if all_done(infer_jobs) and all_done(eval_jobs):
                break

            # Check running inference
            if active_infer is not None:
                # Prefer file-based completion via .done marker: if the output is
                # complete, we treat the job as done and do not inspect the process
                # state here.
                if infer_output_complete(active_infer.infer_path):
                    active_infer.status = "done"
                    active_infer = None

            # Check running evaluation
            if active_eval is not None and active_eval.proc is not None:
                rc = active_eval.proc.poll()
                if rc is not None:
                    if rc == 0:
                        active_eval.status = "done"
                    else:
                        active_eval.status = "failed"
                        logger.error(
                            "[eval] job failed: {} (rc={})", active_eval.eval_path, rc
                        )
                    active_eval.proc = None
                    active_eval = None

            # Start new inference job if none running
            if active_infer is None and args.only in (None, "infer", "eval"):
                next_infer = next(
                    (j for j in infer_jobs if j.status == "pending"), None
                )
                if next_infer is not None:
                    cmd = build_infer_cmd(
                        next_infer.dataset,
                        next_infer.method,
                        next_infer.config_name,
                        next_infer.mas_model,
                        matrix.api_config,
                        infer_cfg,
                        next_infer.agent_config_mode,
                    )
                    logger.info("[infer] {}", " ".join(cmd))
                    if args.dry_run:
                        next_infer.status = "done"
                    else:
                        next_infer.infer_path.parent.mkdir(parents=True, exist_ok=True)
                        next_infer.proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
                        next_infer.status = "running"
                        active_infer = next_infer

            # Start new evaluation job if none running
            if active_eval is None and args.only in (None, "eval"):
                next_eval = next(
                    (j for j in eval_jobs if j.status == "pending" and eval_ready(j)),
                    None,
                )
                if next_eval is not None:
                    cmd = build_eval_cmd(
                        next_eval.dataset,
                        next_eval.method,
                        next_eval.config_name,
                        next_eval.mas_model,
                        matrix.api_config,
                        matrix.eval_model,
                        matrix.eval_protocol,
                        eval_cfg,
                    )
                    logger.info("[eval] {}", " ".join(cmd))
                    if args.dry_run:
                        next_eval.status = "done"
                    else:
                        next_eval.eval_path.parent.mkdir(parents=True, exist_ok=True)
                        next_eval.proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
                        next_eval.status = "running"
                        active_eval = next_eval

            time.sleep(1)
    finally:
        # Final cleanup: best-effort polling of any remaining child processes.
        if active_infer is not None and active_infer.proc is not None:
            rc = active_infer.proc.poll()
            if rc not in (None, 0):
                logger.error(
                    "[infer] job completed but process exited with rc={} for {}",
                    rc,
                    active_infer.infer_path,
                )
            active_infer.proc = None
        if active_eval is not None and active_eval.proc is not None:
            rc = active_eval.proc.poll()
            if rc not in (None, 0):
                logger.error(
                    "[eval] process exited with rc={} for {}",
                    rc,
                    active_eval.eval_path,
                )
            active_eval.proc = None

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="MASLab experiment matrix orchestrator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/experiments.yaml",
        help="Path to experiments YAML config (relative to repo root)",
    )
    parser.add_argument(
        "--only",
        choices=["build", "infer", "eval"],
        default=None,
        help="If set, run only this stage (and its prerequisites if needed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )

    args = parser.parse_args(argv)

    cfg_path = REPO_ROOT / args.config
    if not cfg_path.exists():
        logger.error("Config file not found: {}", cfg_path)
        return 1

    setup_logging()

    matrix = load_yaml_config(cfg_path)
    validate_matrix(matrix)

    defaults_build = matrix.defaults.get("build", {}) if matrix.defaults else {}
    defaults_infer = matrix.defaults.get("inference", {}) if matrix.defaults else {}
    defaults_eval = matrix.defaults.get("evaluate", {}) if matrix.defaults else {}

    build_cfg = BuildConfig(num2sample=defaults_build.get("num2sample"))
    infer_cfg = InferenceConfig(
        require_val=parse_bool_flag(defaults_infer.get("require_val"), False),
        sequential=parse_bool_flag(defaults_infer.get("sequential"), False),
        agent_config_mode=defaults_infer.get("agent_config_mode", "warn"),
    )
    eval_cfg = EvalConfig(
        overwrite=parse_bool_flag(defaults_eval.get("overwrite"), False),
        sequential=parse_bool_flag(defaults_eval.get("sequential"), False),
    )

    # Build datasets once per dataset
    for ds in matrix.datasets:
        dataset = ds.name
        dataset_json = REPO_ROOT / "datasets" / "data" / f"{dataset}.json"
        if args.only in {None, "build", "infer", "eval"}:
            if not dataset_json.exists():
                build_dataset(dataset, build_cfg, dry_run=args.dry_run)
            else:
                logger.info("[build] skip (exists): {}", dataset_json)

    if args.only == "build":
        return 0

    infer_jobs: List[InferJob] = []
    eval_jobs: List[EvalJob] = []

    # Create inference and evaluation jobs
    for ds in matrix.datasets:
        for dm, mas_model in itertools.product(ds.methods, matrix.inference_models):
            dataset = ds.name
            method = dm.method
            config_name = dm.config

            # Per-method inference overrides (e.g., agent_config_mode)
            method_infer_cfg = InferenceConfig(
                require_val=infer_cfg.require_val,
                sequential=infer_cfg.sequential,
                agent_config_mode=infer_cfg.agent_config_mode,
            )
            if dm.inference is not None:
                raw_infer = dm.inference
                if "require_val" in raw_infer:
                    method_infer_cfg.require_val = parse_bool_flag(
                        raw_infer.get("require_val"), infer_cfg.require_val
                    )
                if "sequential" in raw_infer:
                    method_infer_cfg.sequential = parse_bool_flag(
                        raw_infer.get("sequential"), infer_cfg.sequential
                    )
                if "agent_config_mode" in raw_infer:
                    method_infer_cfg.agent_config_mode = raw_infer["agent_config_mode"]

            logger.info("=" * 80)
            logger.info(
                "Dataset={} | Method={} | Config={} | MAS model={}",
                dataset,
                method,
                config_name or "None",
                mas_model,
            )

            infer_path = build_output_path(dataset, method, config_name, mas_model)
            eval_path = infer_path.with_name(
                infer_path.name.replace("infer", f"{matrix.eval_protocol}_eval")
            )

            # Inference jobs
            if args.only in {None, "infer", "eval"}:
                if infer_output_complete(infer_path):
                    logger.info("[infer] skip (done): {}", infer_path)
                elif infer_path.exists():
                    # Legacy or incomplete output without a .done marker; schedule a
                    # new run to bring it to a consistent state.
                    logger.info(
                        "[infer] existing output without .done marker; scheduling rerun: {}",
                        infer_path,
                    )
                    infer_jobs.append(
                        InferJob(
                            dataset=dataset,
                            method=method,
                            config_name=config_name,
                            mas_model=mas_model,
                            infer_path=infer_path,
                            agent_config_mode=method_infer_cfg.agent_config_mode,
                        )
                    )
                else:
                    infer_jobs.append(
                        InferJob(
                            dataset=dataset,
                            method=method,
                            config_name=config_name,
                            mas_model=mas_model,
                            infer_path=infer_path,
                            agent_config_mode=method_infer_cfg.agent_config_mode,
                        )
                    )

            # Evaluation jobs
            if args.only in {None, "eval"}:
                if eval_path.exists() and not eval_cfg.overwrite:
                    logger.info("[eval] skip (exists): {}", eval_path)
                else:
                    if args.only == "eval" and not infer_path.exists():
                        logger.info(
                            "[eval] skip (missing infer output): {} (run inference first or use --only infer/None)",
                            infer_path,
                        )
                    else:
                        eval_jobs.append(
                            EvalJob(
                                dataset=dataset,
                                method=method,
                                config_name=config_name,
                                mas_model=mas_model,
                                infer_path=infer_path,
                                eval_path=eval_path,
                            )
                        )

    return run_scheduler(infer_jobs, eval_jobs, matrix, infer_cfg, eval_cfg, args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
