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

import hydra
from omegaconf import OmegaConf

import pinnstorch
from pinnstorch import utils

INPUT_DIR = Path("/task_data")
OUTPUT_DIR = Path("/task_output")

REQUIRED_FILES = [
    INPUT_DIR / "functions.py",
    INPUT_DIR / "config.yaml",
    INPUT_DIR / "data.mat",
]


def setup(log):
    """Setup output directory structure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(exist_ok=True)
    log.info("Input:  %s", INPUT_DIR)
    log.info("Output: %s", OUTPUT_DIR)


def validate_input(log) -> bool:
    """Check required input files."""
    log.info("Validating input files...")
    for file_path in REQUIRED_FILES:
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
    
    for func_name in ['output_fn', 'initial_fun', 'boundary_functions']:
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


def save_error_report(error: Exception):
    """Save error details."""
    error_path = OUTPUT_DIR / "error.log"
    with open(error_path, 'w') as f:
        f.write("PINN Experiment Failed\n")
        f.write("=" * 60 + "\n\n")
        f.write("Error: %s\n\n" % str(error))
        f.write(traceback.format_exc())


def run():
    log = utils.get_pylogger(__name__)
    
    try:
        log.info("=" * 60)
        log.info("PINN Universal Runner")
        log.info("=" * 60)
        
        setup(log)
        
        if not validate_input(log):
            return 1
        
        user_functions = load_user_functions(INPUT_DIR / "functions.py", log)

        with hydra.initialize_config_dir(
            config_dir=str(INPUT_DIR),
            version_base="1.3"
        ):
            cfg = hydra.compose(config_name="config.yaml")
       
        cfg.paths = cfg.get('paths', {})
        cfg.paths.output_dir = str(OUTPUT_DIR)
        
        data_file = cfg.get('data_file', 'data.mat')
        
        read_data_fn = create_read_data_wrapper(
            user_functions['read_data_fn'], data_file, log
        )
        
        metric_dict, _ = pinnstorch.train(
            cfg=cfg,
            read_data_fn=read_data_fn,
            pde_fn=user_functions['pde_fn'],
            output_fn=user_functions.get('output_fn'),
            boundary_functions=user_functions.get('boundary_functions'),
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
        save_error_report(e)
        return 1


if __name__ == '__main__':
    sys.exit(run())