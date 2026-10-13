"""Exercise the public file runner and measure independent holdout residuals."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.io import loadmat

from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings

from .cases import Array, BenchmarkCase
from .metrics import quality_failures, solution_errors

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(case: BenchmarkCase, device: str, steps: int, seed: int) -> str:
    definition = {
        "contract": 1,
        "config": case.config(device, steps, seed),
        "evaluation": case.config(device, steps, seed, evaluating=True),
        "holdout_points": 2048,
        "holdout_seed": 20261013,
    }
    return hashlib.sha256(
        json.dumps(definition, sort_keys=True).encode() + case.source.read_bytes()
    ).hexdigest()


def launch(input_dir: Path, output_dir: Path, log: Path, checkpoint: Path | None = None) -> float:
    command = [
        sys.executable,
        str(ROOT / "run.py"),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--mode",
        "train" if checkpoint is None else "predict",
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=3600, check=False
        )
    if result.returncode != 0:
        raise RuntimeError(f"Runner failed (exit {result.returncode}); see {log}")
    return time.perf_counter() - started


def holdout_residual(
    case: BenchmarkCase, input_dir: Path, output_dir: Path, checkpoint: Path, device: str
) -> float:
    settings = TaskSettings.load(input_dir / "config.yaml")
    problem = load_problem(input_dir / "functions.py")
    factory = SolverFactory(settings, problem, output_dir, input_dir)
    model = factory.build(predicting=True).model
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(device).eval()
    generator = np.random.default_rng(20261013)
    points = generator.uniform(0.0, 1.0, (2048, case.dimensions + 1))
    points[:, -1] *= 0.5
    columns = [
        torch.tensor(points[:, i : i + 1], dtype=torch.float32, device=device, requires_grad=True)
        for i in range(points.shape[1])
    ]
    with torch.set_grad_enabled(True):
        values = factory.adapter.residuals(model(columns[:-1], columns[-1]), *columns)
        residual = torch.cat([values[name] for name in problem.equations], dim=1)
        return float(residual.detach().square().mean().sqrt().cpu())


def run_trial(
    case: BenchmarkCase, directory: Path, device: str, steps: int, seed: int, smoke: bool
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    task = directory / "task"
    task.mkdir()
    shutil.copyfile(case.source, task / "functions.py")
    config_path = task / "config.yaml"
    config_path.write_text(yaml.safe_dump(case.config(device, steps, seed)), encoding="utf-8")
    training = directory / "training"
    duration = launch(task, training, directory / "training.log")
    checkpoint = training / "checkpoints/final.ckpt"
    state = torch.load(checkpoint, map_location="cpu")
    if state["global_step"] != steps:
        raise RuntimeError(f"Training performed {state['global_step']} steps, expected {steps}")
    config_path.write_text(
        yaml.safe_dump(case.config(device, steps, seed, evaluating=True)), encoding="utf-8"
    )
    prediction = directory / "prediction"
    launch(task, prediction, directory / "prediction.log", checkpoint)
    saved = loadmat(prediction / "predictions.mat")
    spatial = {
        name: np.asarray(saved[name], dtype=np.float64)
        for name in ("x", "y", "z")[: case.dimensions]
    }
    values: Array = np.asarray(saved["u"], dtype=np.float64)
    errors = solution_errors(case, spatial, np.asarray(saved["t"], dtype=np.float64), values)
    errors["pde_rmse"] = holdout_residual(case, task, training, checkpoint, device)
    if not all(math.isfinite(value) for value in errors.values()):
        raise ValueError("Nonfinite benchmark metrics")
    result: dict[str, Any] = {
        "case": case.name,
        "seed": seed,
        "steps": steps,
        "fingerprint": fingerprint(case, device, steps, seed),
        "metrics": errors,
        "training_seconds": duration,
        "quality_failures": [] if smoke else quality_failures(case, errors),
        "quality_checked": not smoke,
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
