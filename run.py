#!/usr/bin/env python3
"""
Universal PINN runner for Docker containers.

Mounted directories:
    /task_data   - User input (functions.py, config.yaml, data.mat)
    /task_output - Results output
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import traceback
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Sequence

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from pinnstorch.data import PointCloudData

INPUT_DIR = Path("/task_data")
OUTPUT_DIR = Path("/task_output")

class Mode(Enum):
    TRAIN = "train"
    RETRAIN = "retrain"
    PREDICT = "predict"

REQUIRED_FILES = {
    Mode.TRAIN: ["functions.py", "config.yaml", "data.mat"],
    Mode.RETRAIN: ["functions.py", "config.yaml", "data.mat", "checkpoint.ckpt"],
    Mode.PREDICT: ["functions.py", "config.yaml", "checkpoint.ckpt"],
}

def setup(log: logging.Logger, output_dir: Path = OUTPUT_DIR) -> None:
    """Setup output directory structure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    log.info("Output: %s", output_dir)


def validate_input(mode: Mode, log: logging.Logger, input_dir: Path = INPUT_DIR) -> bool:
    """Check required input files."""
    log.info("Validating input files...")
    for filename in REQUIRED_FILES[mode]:
        file_path = input_dir / filename
        if not file_path.exists():
            log.error("Missing: %s", file_path)
            return False
        log.info("  Found: %s", file_path.name)
    return True


def load_user_functions(functions_path: Path, log) -> Dict[str, Any]:
    """Load user functions from Python file."""
    log.info("Loading functions: %s", functions_path)
    
    spec = importlib.util.spec_from_file_location("user_functions", functions_path)
    user_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_module)
    
    functions = {}
    
    for func_name in ['read_data_fn', 'pde_fn']:
        if not hasattr(user_module, func_name):
            raise ValueError("Required function '%s' not found" % func_name)
        functions[func_name] = getattr(user_module, func_name)
        log.info("  Loaded: %s", func_name)
    
    for func_name in ['output_fn', 'initial_fun', 'boundary_functions', 'plot_fn']:
        if hasattr(user_module, func_name):
            functions[func_name] = getattr(user_module, func_name)
            log.info("  Loaded: %s", func_name)
        else:
            log.info("  Not found: %s", func_name)
    
    return functions


def create_read_data_wrapper(
    user_read_data_fn: Callable[[dict[str, Any]], dict[str, Any]],
    data_file_name: str,
    log: logging.Logger,
    input_dir: Path = INPUT_DIR,
) -> Callable[[], PointCloudData]:
    """Wrap user read_data_fn to load .mat file automatically."""
    
    import pinnstorch

    def wrapped_read_data_fn() -> PointCloudData:
        log.info("Loading data: %s", data_file_name)
        raw_data = pinnstorch.utils.load_data(input_dir, data_file_name)
        user_result = user_read_data_fn(raw_data)
        
        return pinnstorch.data.PointCloudData(
            spatial=user_result['spatial'],
            time=user_result['time'],
            solution=user_result['solution']
        )
    
    return wrapped_read_data_fn

def create_plot_func_wrapper(user_plot_fn):
    import pinnstorch

    if user_plot_fn is None:
        return None
    
    def wrapped_plot_func(mesh, preds, train_datasets, val_dataset, filename):
        preds_np = {k: v.detach().cpu().numpy() for k, v in preds.items()}
        user_plot_fn(mesh, preds_np, train_datasets, val_dataset)
        pinnstorch.utils.savefig(filename + "/fig", False)
    
    return wrapped_plot_func

def save_error_report(error: Exception, output_dir: Path = OUTPUT_DIR) -> None:
    """Save error details."""
    error_path = output_dir / "error.log"
    with open(error_path, 'w') as f:
        f.write("PINN Experiment Failed\n")
        f.write("=" * 60 + "\n\n")
        f.write("Error: %s\n\n" % str(error))
        f.write(traceback.format_exc())

def setup_config(
    cfg: DictConfig, output_dir: Path = OUTPUT_DIR
) -> DictConfig: 
    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    
    cfg.mesh = {}
    cfg.mesh._target_ = "pinnstorch.data.PointCloud"
    
    for item in cfg.train_datasets:
        for key in item:
            item[key]["_partial_"] = True
        
    for group in ["val_dataset", "pred_dataset"]:
        if group in cfg:
            for item in cfg[group]:
                for key in item:
                    item[key]["_partial_"] = True
    
    cfg.model._target_ = "pinnstorch.models.PINNModule"    
    cfg.model._partial_ = True
            
    cfg.model.optimizer._partial_ = True
    if cfg.model.scheduler:
        cfg.model.scheduler._partial_ = True
    
    if cfg.model.optimizer._target_ == "torch.optim.Adam":
        cfg.model.optimizer.capturable = False
    
    cfg.trainer._target_ = "lightning.pytorch.Trainer"
    
    cfg.model.loss_fn = "mse"
    
    cfg.data = {}
    cfg.data._target_ = "pinnstorch.data.PINNDataModule"

    if cfg.net._target_ == "pinnstorch.models.FCN":
        cfg.net._partial_ = True
    
    cfg.train = True
    cfg.val = True
    cfg.test = False
    
    cfg.paths = cfg.get('paths', {})
    cfg.paths.output_dir = str(output_dir)
    
    return cfg
    
class RunnerArguments(argparse.Namespace):
    """Arguments accepted by the container entry point."""

    mode: str
    checkpoint: str | None
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
        type=str,
        default=None,
        help="Имя файла с весами ранее обученной модели",
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="Directory with task files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for results")
    return parser.parse_args(argv, namespace=RunnerArguments())


def run(mode: Mode, input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> int:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

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
        
        if not validate_input(mode, log, input_dir):
            return 1
        
        user_functions = load_user_functions(input_dir / "functions.py", log)

        with initialize_config_dir(
            config_dir=str(input_dir),
            version_base="1.3"
        ):
            cfg = hydra.compose(config_name="config.yaml")
        
        package_path = os.path.dirname((pinnstorch.__file__))
        train_conf_path = os.path.join(package_path, "conf")
        
        with initialize_config_dir(version_base="1.3", config_dir=train_conf_path):
            # 3. Собираем конфиг. Сюда подтянутся все файлы из секции defaults
            train_cfg = compose(config_name="train") 
        
        OmegaConf.set_struct(train_cfg, False)
        # print("train_cfg")
        # print(OmegaConf.to_yaml(train_cfg))
        
        cfg = OmegaConf.merge(train_cfg, cfg)
        
        # print("cfg after merge")
        # print(OmegaConf.to_yaml(cfg))
        
        read_data_fn = create_read_data_wrapper(
            user_functions['read_data_fn'], 'data.mat', log, input_dir
        )
        
        cfg = setup_config(cfg, output_dir)
        log.info(f"running mode: {mode.name}({mode.value})")
        # print("cfg after setup")
        # print(OmegaConf.to_yaml(cfg))
        
        plot_func = create_plot_func_wrapper(user_functions.get('plot_fn'))
        
        metric_dict, _ = pinnstorch.train(
            cfg=cfg,
            pde_fn=user_functions['pde_fn'],
            mode=mode.value,
            read_data_fn=read_data_fn,
            output_fn=user_functions.get('output_fn'),
            boundary_functions=user_functions.get('boundary_functions'),
            plot_func=plot_func,
            checkpoint=input_dir / "checkpoint.ckpt",
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
    return run(Mode(args.mode), input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == '__main__':
    sys.exit(main())
