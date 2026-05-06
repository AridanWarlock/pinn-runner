#!/usr/bin/env python3
"""
Universal PINN runner for Docker containers.

Mounted directories:
    /task_data   - User input (functions.py, config.yaml, data.mat)
    /task_output - Results output
"""

import sys
import traceback
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import argparse

import os
import time
import hydra
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from omegaconf import DictConfig
from enum import Enum

import pinnstorch
from pinnstorch import utils

INPUT_DIR = Path("/task_data")
OUTPUT_DIR = Path("/task_output")

class Mode(Enum):
    TRAIN = "train"
    RETRAIN = "retrain"
    PREDICT = "predict"

FUNC_FILE = INPUT_DIR / "functions.py"
CONF_FILE = INPUT_DIR / "config.yaml"
DATA_FILE = INPUT_DIR / "data.mat"
CHECKPOINT_FILE = INPUT_DIR / "checkpoint.ckpt"

REQUIRED_FILES = {
    Mode.TRAIN: [FUNC_FILE, CONF_FILE, DATA_FILE],
    Mode.RETRAIN: [FUNC_FILE, CONF_FILE, DATA_FILE, CHECKPOINT_FILE],
    Mode.PREDICT: [FUNC_FILE, CONF_FILE, CHECKPOINT_FILE],
}

def setup(log):
    """Setup output directory structure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
    log.info("Input:  %s", INPUT_DIR)
    log.info("Output: %s", OUTPUT_DIR)


def validate_input(mode, log) -> bool:
    """Check required input files."""
    log.info("Validating input files...")
    for file_path in REQUIRED_FILES[mode]:
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


def create_read_data_wrapper(user_read_data_fn, data_file_name: str, log):
    """Wrap user read_data_fn to load .mat file automatically."""
    
    def wrapped_read_data_fn():
        log.info("Loading data: %s", data_file_name)
        raw_data = pinnstorch.utils.load_data(INPUT_DIR, data_file_name)
        user_result = user_read_data_fn(raw_data)
        
        return pinnstorch.data.PointCloudData(
            spatial=user_result['spatial'],
            time=user_result['time'],
            solution=user_result['solution']
        )
    
    return wrapped_read_data_fn

def create_plot_func_wrapper(user_plot_fn):
    if user_plot_fn == None:
        return None
    
    def wrapped_plot_func(mesh, preds, train_datasets, val_dataset, filename):
        preds_np = {k: v.detach().cpu().numpy() for k, v in preds.items()}
        user_plot_fn(mesh, preds_np, train_datasets, val_dataset)
        pinnstorch.utils.savefig(filename + "/fig", False)
    
    return wrapped_plot_func

def save_error_report(error: Exception):
    """Save error details."""
    error_path = OUTPUT_DIR / "error.log"
    with open(error_path, 'w') as f:
        f.write("PINN Experiment Failed\n")
        f.write("=" * 60 + "\n\n")
        f.write("Error: %s\n\n" % str(error))
        f.write(traceback.format_exc())

def setup_config(
    cfg: DictConfig
) -> DictConfig: 
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
    cfg.trainer.accelerator = "gpu" 
    cfg.devices = [0]
    
    cfg.model.loss_fn = "mse"
    
    cfg.data = {}
    cfg.data._target_ = "pinnstorch.data.PINNDataModule"

    if cfg.net._target_ == "pinnstorch.models.FCN":
        cfg.net._partial_ = True
    
    cfg.train = True
    cfg.val = True
    cfg.test = False
    
    cfg.seed = int(time.time())
    
    cfg.paths = cfg.get('paths', {})
    cfg.paths.output_dir = str(OUTPUT_DIR)
    
    return cfg
    
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='Режим работы (train, retrain, predict)')
parser.add_argument('--checkpoint', type=str, default=None, help='Имя файла с весами ранее обученной модели')
args = parser.parse_args()

def run():
    log = utils.get_pylogger(__name__)
    
    try:        
        log.info("=" * 60)
        log.info("PINN Universal Runner")
        log.info("=" * 60)
        
        mode = Mode(args.mode)
        
        match mode:
            case Mode.TRAIN:
                pass
            case Mode.RETRAIN:
                pass
            case Mode.PREDICT:
                pass
            case _:
                raise ValueError(f"Недопустимый режим {args.mode}")
        
        setup(log)
        
        if not validate_input(mode, log):
            return 1
        
        user_functions = load_user_functions(FUNC_FILE, log)

        with initialize_config_dir(
            config_dir=str(INPUT_DIR),
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
            user_functions['read_data_fn'], 'data.mat', log
        )
        
        cfg = setup_config(cfg)
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
            checkpoint=CHECKPOINT_FILE,
        )
        
        log.info("=" * 60)
        log.info("Training completed")
        for k, v in metric_dict.items():
            if isinstance(v, (int, float)):
                log.info("  %s: %.6f", k, v)
        log.info("Results: %s", OUTPUT_DIR)
        log.info("=" * 60)
        
        return 0
        
    except Exception as e:
        log.error("=" * 60)
        log.error("Training failed")
        log.error("Error: %s", str(e))
        log.error(traceback.format_exc())
        log.error("=" * 60)
        
        print(f"Running error: {str(e)}", file=sys.stderr)
        
        save_error_report(e)
        return 1


if __name__ == '__main__':
    sys.exit(run())