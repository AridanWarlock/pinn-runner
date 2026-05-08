import argparse
import importlib.util
import logging
import os
import sys
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Callable

import hydra
import scipy
import torch
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig
from omegaconf import OmegaConf

import pinnstorch as pt
from pinnstorch import utils

INPUT_DIR = Path("/task_data")
OUTPUT_DIR = Path("/task_output")

FUNC_FILE = INPUT_DIR / "functions.py"
CONF_FILE = INPUT_DIR / "config.yaml"
DATA_FILE = INPUT_DIR / "data.mat"
CHECKPOINT_FILE = INPUT_DIR / "checkpoint.ckpt"


class Mode(Enum):
    TRAIN = "train"
    RETRAIN = "retrain"
    PREDICT = "predict"


REQUIRED_FILES = {
    Mode.TRAIN: [FUNC_FILE, CONF_FILE, DATA_FILE],
    Mode.RETRAIN: [FUNC_FILE, CONF_FILE, DATA_FILE, CHECKPOINT_FILE],
    Mode.PREDICT: [FUNC_FILE, CONF_FILE, DATA_FILE, CHECKPOINT_FILE],
}

REQUIRED_FUNCTIONS = ['read_data_fn', 'pde_fn', 'boundary_functions']
OPTIONAL_FUNCTIONS = ['output_fn', 'initial_fun', 'plot_fn']

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='Режим работы (train, retrain, predict)')
args = parser.parse_args()

log: logging.Logger = utils.get_pylogger(__name__, OUTPUT_DIR / "log.log")
mode = Mode(args.mode)


def setup_dirs():
    """Setup output directory structure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
    log.info("Input:  %s", INPUT_DIR)
    log.info("Output: %s", OUTPUT_DIR)


def validate_input():
    """Check required input files."""
    log.info("Validating input files...")
    for file_path in REQUIRED_FILES[mode]:
        if not file_path.exists():
            log.error("Missing: %s", file_path)
            raise ValueError(f"Missing required file: {file_path}")


def load_user_functions() -> Dict[str, Any]:
    """Load user functions from Python file."""
    log.info("Loading functions: %s", FUNC_FILE)

    spec = importlib.util.spec_from_file_location("user_functions", FUNC_FILE)
    user_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_module)

    functions = {}

    for func_name in REQUIRED_FUNCTIONS:
        if not hasattr(user_module, func_name):
            raise ValueError("Required function '%s' not found" % func_name)
        functions[func_name] = getattr(user_module, func_name)
        log.info("  Loaded: %s", func_name)

    for func_name in OPTIONAL_FUNCTIONS:
        if hasattr(user_module, func_name):
            functions[func_name] = getattr(user_module, func_name)
            log.info("  Loaded: %s", func_name)
        else:
            log.info("  Not found: %s", func_name)

    return functions


def create_read_data_wrapper(user_read_data_fn: Callable[[Dict], Dict[str, Any]]):
    """Wrap user read_data_fn to load .mat file automatically."""

    def wrapped_read_data_fn():
        log.info("Loading data.mat")
        raw_data = scipy.io.loadmat(str(DATA_FILE))
        user_result = user_read_data_fn(raw_data)

        required_keys = {"spatial", "time", "solution"}
        missing = required_keys - user_result.keys()
        if missing:
            raise KeyError(f"Отсутствуют обязательные ключи: {missing}")

        return pt.PointCloudData(
            spatial=user_result['spatial'],
            time=user_result['time'],
            solution=user_result['solution']
        )

    return wrapped_read_data_fn


def create_plot_func_wrapper(user_plot_fn):
    if user_plot_fn is None:
        return None

    def wrapped_plot_func(
            mesh: pt.MeshBase,
            preds: Dict[str, torch.Tensor],
            train_datasets,
            val_dataset,
            filename,
    ):
        preds_np = {k: v.detach().cpu().numpy() for k, v in preds.items()}
        user_plot_fn(mesh, preds_np, train_datasets, val_dataset)
        pt.utils.savefig(filename + "/fig", False)

    return wrapped_plot_func


def setup_config() -> DictConfig:
    package_path = os.path.dirname(pt.__file__)
    default_conf_path = os.path.join(package_path, "conf")
    with initialize_config_dir(
            config_dir=default_conf_path,
            version_base="1.3"
    ):
        default_cfg = compose("train")
    OmegaConf.set_struct(default_cfg, False)

    with initialize_config_dir(
            config_dir=str(INPUT_DIR),
            version_base="1.3"
    ):
        cfg = hydra.compose(config_name="config")
    OmegaConf.set_struct(cfg, False)

    cfg = OmegaConf.merge(default_cfg, cfg)
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
    if mode in (Mode.TRAIN, Mode.RETRAIN):
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.devices = [0]
    else:
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1

    cfg.model.loss_fn = "mse"

    cfg.data = {}
    cfg.data._target_ = "pinnstorch.data.PINNDataModule"

    if cfg.net._target_ == "pinnstorch.models.FCN":
        cfg.net._partial_ = True

    cfg.train = True
    cfg.val = True
    cfg.test = False

    cfg.seed = int(time.time())

    cfg.paths = {}
    cfg.paths.output_dir = str(OUTPUT_DIR)

    return cfg


def run() -> Dict[str, Any]:
    validate_input()
    setup_dirs()
    cfg = setup_config()

    user_functions = load_user_functions()
    read_data_fn = create_read_data_wrapper(
        user_functions['read_data_fn']
    )
    plot_func = create_plot_func_wrapper(user_functions.get('plot_fn'))

    metric_dict, _ = pt.train(
        cfg=cfg,
        log=log,
        pde_fn=user_functions['pde_fn'],
        mode=mode.value,
        read_data_fn=read_data_fn,
        output_fn=user_functions.get('output_fn'),
        boundary_functions=user_functions.get('boundary_functions'),
        plot_func=plot_func,
        checkpoint=CHECKPOINT_FILE,
    )
    return metric_dict


if __name__ == '__main__':
    log.info("=" * 60)
    log.info("PINN Universal Runner")
    log.info("=" * 60)

    try:
        metric_dict = run()

        log.info("=" * 60)
        log.info(f"{mode.name} completed")
        for k, v in metric_dict.items():
            if isinstance(v, (int, float)):
                log.info("  %s: %.6f", k, v)
        log.info("Results: %s", OUTPUT_DIR)
        log.info("=" * 60)
    except Exception as e:
        log.error("=" * 60)
        log.error(f"{mode.name} failed")
        log.error("Error: %s", str(e))
        log.error(traceback.format_exc())
        log.error("=" * 60)

        print(f"Running error: {str(e)}", file=sys.stderr)
        sys.exit(1)
