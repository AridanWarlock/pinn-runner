#!/usr/bin/env python3
"""
Universal PINN runner for Docker containers.

Mounted directories:
    /task_data   - User input (functions.py, config.yaml)
    /task_output - Results output
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import traceback
from collections.abc import Callable, Sequence
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from pinnstorch.contracts import (
        ArrayOutputs,
        PlotFunction,
        SamplerDataset,
        TensorOutputs,
        UserFunctions,
    )
    from pinnstorch.data.mesh.mesh import MeshBase

INPUT_DIR = Path("/task_data")
OUTPUT_DIR = Path("/task_output")


class Mode(Enum):
    TRAIN = "train"
    RETRAIN = "retrain"
    PREDICT = "predict"


REQUIRED_FILES = {
    Mode.TRAIN: ["functions.py", "config.yaml"],
    Mode.RETRAIN: ["functions.py", "config.yaml", "checkpoint.ckpt"],
    Mode.PREDICT: ["functions.py", "config.yaml", "checkpoint.ckpt"],
}


def setup(log: logging.Logger, output_dir: Path = OUTPUT_DIR) -> None:
    """Setup output directory structure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    log.info("Output: %s", output_dir)


def validate_input(
    mode: Mode,
    log: logging.Logger,
    input_dir: Path = INPUT_DIR,
    checkpoint: Path | None = None,
) -> bool:
    """Check required input files."""
    log.info("Validating input files...")
    for filename in REQUIRED_FILES[mode]:
        file_path = input_dir / filename
        if filename == "checkpoint.ckpt" and checkpoint is not None:
            file_path = checkpoint
        if not file_path.exists():
            log.error("Missing: %s", file_path)
            return False
        log.info("  Found: %s", file_path.name)
    return True


def load_user_functions(functions_path: Path, log: logging.Logger) -> UserFunctions:
    """Load user functions from Python file."""
    log.info("Loading functions: %s", functions_path)

    spec = importlib.util.spec_from_file_location("user_functions", functions_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load Python functions from '{functions_path}'")
    user_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_module)

    functions: dict[str, Any] = {}

    for func_name in ["pde_fn", "initial_conditions"]:
        if not callable(getattr(user_module, func_name, None)):
            raise ValueError("Required function '%s' not found" % func_name)
        functions[func_name] = getattr(user_module, func_name)
        log.info("  Loaded: %s", func_name)

    for func_name in ["output_fn", "boundary_functions", "boundary_values", "plot_fn"]:
        if hasattr(user_module, func_name):
            functions[func_name] = getattr(user_module, func_name)
            log.info("  Loaded: %s", func_name)
        else:
            log.info("  Not found: %s", func_name)

    return cast("UserFunctions", functions)


def create_plot_func_wrapper(
    user_plot_fn: PlotFunction | None,
) -> (
    Callable[
        [MeshBase, TensorOutputs, Sequence[SamplerDataset], SamplerDataset | None, str],
        None,
    ]
    | None
):
    import pinnstorch

    if user_plot_fn is None:
        return None

    def wrapped_plot_func(
        mesh: MeshBase,
        preds: TensorOutputs,
        train_datasets: Sequence[SamplerDataset],
        val_dataset: SamplerDataset | None,
        filename: str,
    ) -> None:
        preds_np: ArrayOutputs = {k: v.detach().cpu().numpy() for k, v in preds.items()}
        user_plot_fn(mesh, preds_np, train_datasets, val_dataset)
        pinnstorch.utils.savefig(filename + "/fig", False)

    return wrapped_plot_func


def save_error_report(error: Exception, output_dir: Path = OUTPUT_DIR) -> None:
    """Save error details."""
    error_path = output_dir / "error.log"
    with open(error_path, "w") as f:
        f.write("PINN Experiment Failed\n")
        f.write("=" * 60 + "\n\n")
        f.write("Error: %s\n\n" % str(error))
        f.write(traceback.format_exc())


def setup_config(cfg: DictConfig, output_dir: Path = OUTPUT_DIR) -> DictConfig:
    from omegaconf import DictConfig, OmegaConf

    OmegaConf.set_struct(cfg, False)

    if not cfg.get("domain"):
        raise ValueError(
            "config.yaml must define domain (x_interval, t_interval, x_points, t_points)"
        )
    if not isinstance(cfg.domain, DictConfig):
        raise ValueError("domain must be a mapping of interval bounds and point counts")
    if not any(
        item.get("_target_") == "pinnstorch.data.InitialCondition"
        for group in cfg.train_datasets
        for item in group.values()
    ):
        raise ValueError("train_datasets must include InitialCondition")
    boundary_targets = [
        item.get("_target_")
        for group in cfg.train_datasets
        for item in group.values()
        if "BoundaryCondition" in str(item.get("_target_", ""))
    ]
    if "pinnstorch.data.BoundaryConditions" in boundary_targets and len(boundary_targets) != 1:
        raise ValueError(
            "Put all boundary conditions in one BoundaryConditions dataset; do not mix boundary APIs"
        )
    cfg.mesh = {}
    cfg.mesh._target_ = "pinnstorch.data.PointCloud"

    for item in cfg.train_datasets:
        for key in item:
            item[key]["_partial_"] = True

    for group in ["val_dataset", "pred_dataset"]:
        if cfg.get(group) is not None:
            for item in cfg[group]:
                for key in item:
                    item[key]["_partial_"] = True

    cfg.model._target_ = "pinnstorch.models.PINNModule"
    cfg.model._partial_ = True

    cfg.model.optimizer._partial_ = True
    if cfg.model.get("scheduler"):
        cfg.model.scheduler._partial_ = True

    if cfg.model.optimizer._target_ == "torch.optim.Adam":
        cfg.model.optimizer.capturable = False

    cfg.trainer._target_ = "lightning.pytorch.Trainer"

    cfg.model.loss_fn = "mse"

    cfg.data = cfg.get("data", {})
    cfg.data._target_ = "pinnstorch.data.PINNDataModule"

    if cfg.net._target_ == "pinnstorch.models.FCN":
        cfg.net._partial_ = True

    cfg.train = True
    cfg.val = cfg.get("val_dataset") is not None
    cfg.test = False

    cfg.paths = cfg.get("paths", {})
    cfg.paths.output_dir = str(output_dir)
    cfg.paths.work_dir = str(Path.cwd())

    return cfg


class RunnerArguments(argparse.Namespace):
    """Arguments accepted by the container entry point."""

    mode: str
    checkpoint: Path | None
    input_dir: Path
    output_dir: Path


def parse_args(argv: Sequence[str] | None = None) -> RunnerArguments:
    """Parse CLI arguments only when explicitly called."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=[mode.value for mode in Mode],
        help="Режим работы (train, retrain, predict)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Имя файла с весами ранее обученной модели",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=INPUT_DIR, help="Directory with task files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for results"
    )
    return parser.parse_args(argv, namespace=RunnerArguments())


def run(
    mode: Mode,
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    checkpoint: Path | None = None,
) -> int:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    checkpoint_path = (input_dir / (checkpoint or Path("checkpoint.ckpt"))).resolve()
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf

    import pinnstorch
    from pinnstorch import utils

    log = utils.get_pylogger(__name__)

    try:
        log.info("=" * 60)
        log.info("PINN Universal Runner")
        log.info("=" * 60)

        log.info("Input:  %s", input_dir)
        setup(log, output_dir)
        utils.get_pylogger("pinnstorch.train", output_path=output_dir / "log.log")

        if not validate_input(mode, log, input_dir, checkpoint_path):
            return 1

        user_functions = load_user_functions(input_dir / "functions.py", log)

        with initialize_config_dir(config_dir=str(input_dir), version_base="1.3"):
            cfg = hydra.compose(config_name="config.yaml")

        package_path = os.path.dirname((pinnstorch.__file__))
        train_conf_path = os.path.join(package_path, "conf")

        with initialize_config_dir(version_base="1.3", config_dir=train_conf_path):
            # 3. Собираем конфиг. Сюда подтянутся все файлы из секции defaults
            train_cfg = compose(config_name="train")

        OmegaConf.set_struct(train_cfg, False)

        cfg = cast("DictConfig", OmegaConf.merge(train_cfg, cfg))

        cfg = setup_config(cfg, output_dir)
        prediction = cfg.get("prediction", {})
        if not isinstance(prediction, (DictConfig, dict)):
            raise ValueError("prediction must be a mapping")
        unknown_prediction = set(prediction) - {"parameters", "allow_extrapolation"}
        if unknown_prediction:
            raise ValueError(f"Unknown prediction options: {sorted(unknown_prediction)}")
        if not isinstance(prediction.get("allow_extrapolation", False), bool):
            raise ValueError("prediction.allow_extrapolation must be true or false")
        query_values = prediction.get("parameters", {})
        if not isinstance(query_values, (DictConfig, dict)):
            raise ValueError("prediction.parameters must map names to lists of values")
        if "parameter_values" in cfg.domain:
            raise ValueError("Set explicit values under prediction.parameters, not domain")
        resolved_query = (
            cast("dict[str, Any]", OmegaConf.to_container(query_values, resolve=True))
            if isinstance(query_values, DictConfig)
            else query_values
        )
        read_data_fn = partial(
            pinnstorch.data.create_interval_data,
            **cast("dict[str, Any]", OmegaConf.to_container(cfg.domain, resolve=True)),
            parameter_values=resolved_query if mode == Mode.PREDICT else None,
        )
        from pinnstorch.parameter_functions import bind_parameters

        spatial_dimension = 1 + int("y_interval" in cfg.domain) + int("z_interval" in cfg.domain)
        parameters = cfg.domain.get("parameters")
        if parameters is not None and not isinstance(parameters, DictConfig):
            raise ValueError("domain.parameters must be a mapping of named intervals")
        parameters = parameters or {}
        from pinnstorch.config_validation import validate_task_settings

        validate_task_settings(cfg, spatial_dimension, tuple(parameters))
        if parameters and (
            cfg.model.get("extra_variables")
            or cfg.model.get("jit_compile")
            or cfg.net.get("discrete")
        ):
            raise ValueError(
                "Parameterized tasks require continuous mode without inverse variables or JIT"
            )
        user_functions = bind_parameters(user_functions, spatial_dimension, tuple(parameters))
        OmegaConf.save(cfg, output_dir / "config.resolved.yaml", resolve=True)
        log.info(f"running mode: {mode.name}({mode.value})")

        plot_func = create_plot_func_wrapper(user_functions.get("plot_fn"))

        metric_dict, _ = pinnstorch.train(
            cfg=cfg,
            pde_fn=user_functions["pde_fn"],
            initial_conditions=user_functions["initial_conditions"],
            mode=mode.value,
            read_data_fn=read_data_fn,
            output_fn=user_functions.get("output_fn"),
            boundary_functions=user_functions.get("boundary_functions"),
            boundary_values=user_functions.get("boundary_values"),
            plot_func=plot_func,
            checkpoint=checkpoint_path,
        )

        log.info("=" * 60)
        log.info("Training completed")
        for k, v in metric_dict.items():
            if isinstance(v, (int, float)):
                log.info("  %s: %.6f", k, v)
        log.info("Results: %s", output_dir)
        log.info("=" * 60)

        return 0

    except Exception as e:
        log.error("=" * 60)
        log.error("Training failed")
        log.error("Error: %s", str(e))
        log.error(traceback.format_exc())
        log.error("=" * 60)

        print(f"Running error: {str(e)}", file=sys.stderr)

        save_error_report(e, output_dir)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Validate CLI arguments before loading the training dependencies."""
    args = parse_args(argv)
    return run(
        Mode(args.mode),
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    sys.exit(main())
