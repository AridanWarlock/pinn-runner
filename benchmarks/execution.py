"""Exercise the public file runner and measure independent holdout residuals."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.io import loadmat

from pinnstorch.runner.checkpoint_io import load_checkpoint
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings

from .cases import Array, BenchmarkCase
from .diagnostics import residual_time_bins, worst_time_error
from .metrics import quality_failures, solution_errors, temporal_errors
from .performance import prediction_speed

ROOT = Path(__file__).resolve().parents[1]
WORKING_DIRECTORY = Path(os.environ.get("PINN_WORKING_DIRECTORY", str(ROOT)))


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
            command,
            cwd=WORKING_DIRECTORY,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=3600,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Runner failed (exit {result.returncode}); see {log}")
    return time.perf_counter() - started


@dataclass(frozen=True)
class CheckpointEvaluation:
    residual_rmse: float
    residual_bins: list[dict[str, float | int | None]]
    performance: dict[str, float | int]


def evaluate_checkpoint(
    case: BenchmarkCase, input_dir: Path, output_dir: Path, checkpoint: Path, device: str
) -> CheckpointEvaluation:
    settings = TaskSettings.load(input_dir / "config.yaml")
    problem = load_problem(input_dir / "functions.py")
    factory = SolverFactory(settings, problem, output_dir, input_dir)
    objects = factory.build(predicting=True)
    model = objects.model
    state = load_checkpoint(checkpoint)
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(device).eval()
    points = holdout_inputs(settings)
    lower, upper = settings.domain.time.interval
    columns = [
        torch.tensor(points[:, i : i + 1], dtype=torch.float32, device=device, requires_grad=True)
        for i in range(points.shape[1])
    ]
    with torch.set_grad_enabled(True):
        # Independent validation: differentiate the physical field, bypassing
        # any prepared derivative rule used to accelerate training.
        physical = dict(model(columns[:-1], columns[-1]))
        values = factory.adapter.residuals(physical, *columns)
        residual = torch.cat([values[name] for name in problem.equations], dim=1)
        error = float(residual.detach().square().mean().sqrt().cpu())
    bins = residual_time_bins(
        points[:, -1:], residual.detach().cpu().numpy().astype(np.float64), (lower, upper)
    )
    del values, residual, columns
    spatial, times, _ = objects.mesh.flatten_mesh(None)
    features = [
        torch.tensor(spatial[:, i : i + 1], dtype=torch.float32, device=device)
        for i in range(spatial.shape[1])
    ]
    clock = torch.tensor(times.copy(), dtype=torch.float32, device=device)
    return CheckpointEvaluation(
        error,
        bins,
        prediction_speed(model, features, clock, settings.training.batch_size or len(clock)),
    )


def holdout_inputs(settings: TaskSettings, count: int = 2048, seed: int = 20261013) -> Array:
    """Sample every named input in physical bounds, including family parameters."""
    axes = (
        *settings.domain.space.values(),
        *settings.domain.parameters.values(),
        settings.domain.time,
    )
    if count < 1:
        raise ValueError("Holdout point count must be positive")
    unit = np.random.default_rng(seed).uniform(0.0, 1.0, (count, len(axes)))
    for index, axis in enumerate(axes):
        lower, upper = axis.interval
        unit[:, index] = lower + (upper - lower) * unit[:, index]
    return unit


def prepare_task(case: BenchmarkCase, directory: Path, device: str, steps: int, seed: int) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    task = directory / "task"
    task.mkdir()
    shutil.copyfile(case.source, task / "functions.py")
    (task / "config.yaml").write_text(
        yaml.safe_dump(case.config(device, steps, seed)), encoding="utf-8"
    )
    return task


def validate_refinement(checkpoint: dict[str, Any], steps: int, device: str) -> None:
    phase = checkpoint.get("refinement_state") or {}
    if steps == 0:
        if phase:
            raise RuntimeError("Unexpected refinement in Adam-only benchmark")
        return
    if (
        phase.get("kind") != "lbfgs"
        or phase.get("version") != 1
        or phase.get("completed") is not True
        or phase.get("metrics", {}).get("lbfgs_steps") != steps
        or str(phase.get("device", "")).split(":")[0] != device
    ):
        raise RuntimeError("Refinement stage incomplete or executed on the wrong backend")


def finish_trial(
    case: BenchmarkCase,
    directory: Path,
    task: Path,
    checkpoint: Path,
    device: str,
    steps: int,
    seed: int,
    duration: float,
    smoke: bool,
) -> dict[str, Any]:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Training duration must be finite and positive")
    state = load_checkpoint(checkpoint)
    if state["global_step"] != steps:
        raise RuntimeError(f"Training performed {state['global_step']} steps, expected {steps}")
    refinement = case.config(device, steps, seed)["training"].get("refinement", {})
    validate_refinement(state, int(refinement.get("steps", 0)), device)
    (task / "config.yaml").write_text(
        yaml.safe_dump(case.config(device, steps, seed, evaluating=True)), encoding="utf-8"
    )
    prediction = directory / "prediction"
    prediction_duration = launch(task, prediction, directory / "prediction.log", checkpoint)
    saved = loadmat(prediction / "predictions.mat")
    spatial = {
        name: np.asarray(saved[name], dtype=np.float64)
        for name in ("x", "y", "z")[: case.dimensions]
    }
    values: Array = np.asarray(saved["u"], dtype=np.float64)
    errors = solution_errors(case, spatial, np.asarray(saved["t"], dtype=np.float64), values)
    evaluation = evaluate_checkpoint(case, task, directory, checkpoint, device)
    errors["pde_rmse"] = evaluation.residual_rmse
    performance = evaluation.performance
    temporal = temporal_errors(case, spatial, np.asarray(saved["t"], dtype=np.float64), values)
    if not all(math.isfinite(value) for value in errors.values()):
        raise ValueError("Nonfinite benchmark metrics")
    performance.update(
        training_seconds=duration,
        training_steps_per_second=steps / duration,
        prediction_seconds=prediction_duration,
        prediction_points_per_second=values.size / prediction_duration,
    )
    memory_file = checkpoint.parent.parent / "training.performance.json"
    if memory_file.exists():
        performance.update(json.loads(memory_file.read_text(encoding="utf-8")))
    if performance.get("lbfgs_steps", 0):
        performance["optimizer_calls_per_second"] = (steps + performance["lbfgs_steps"]) / duration
        performance["total_model_evaluations"] = (
            performance["adam_model_evaluations"] + performance["lbfgs_closure_evaluations"]
        )
        performance["total_residual_points"] = (
            performance["adam_residual_points"] + performance["lbfgs_residual_points"]
        )
    result: dict[str, Any] = {
        "case": case.name,
        "seed": seed,
        "steps": steps,
        "fingerprint": fingerprint(case, device, steps, seed),
        "constraint_kind": state.get("hyper_parameters", {}).get("constraint_kind", "soft"),
        "hard_execution": state.get("hyper_parameters", {}).get("hard_execution", "eager"),
        "hard_lifting": state.get("hyper_parameters", {}).get("hard_lifting", "recursive"),
        "backward_mode": state.get("hyper_parameters", {}).get("backward_mode", "all"),
        "pde_validation_backend": "physical_autograd",
        "refinement_recipe": state.get("hyper_parameters", {}).get("refinement_settings")
        or {"steps": 0},
        "refinement_device": state.get("refinement_state", {}).get("device"),
        "sampling_recipe": {
            **case.config(device, steps, seed)["training"]["collocation"],
            "version": 1,
        },
        "schedule_recipe": state.get("hyper_parameters", {}).get("rate_settings")
        or {"kind": "constant"},
        "training_recipe": state.get("hyper_parameters", {}).get("loss_settings")
        or {"kind": "equal"},
        "metrics": errors,
        "temporal_metrics": temporal,
        "diagnostics": {
            **worst_time_error(temporal),
            "pde_time_bins": evaluation.residual_bins,
            "holdout_seed": 20261013,
            "holdout_points": 2048,
            "version": 1,
        },
        "training_seconds": duration,
        "performance": performance,
        "quality_failures": [] if smoke else quality_failures(case, errors),
        "quality_checked": not smoke,
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_trial(
    case: BenchmarkCase, directory: Path, device: str, steps: int, seed: int, smoke: bool
) -> dict[str, Any]:
    task = prepare_task(case, directory, device, steps, seed)
    training = directory / "training"
    duration = launch(task, training, directory / "training.log")
    return finish_trial(
        case,
        directory,
        task,
        training / "checkpoints/final.ckpt",
        device,
        steps,
        seed,
        duration,
        smoke,
    )


def measure_existing(
    case: BenchmarkCase,
    previous: Path,
    directory: Path,
    device: str,
    steps: int,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    original = json.loads((previous / "result.json").read_text(encoding="utf-8"))
    if original["fingerprint"] != fingerprint(case, device, steps, seed):
        raise ValueError("Existing training task/budget/seed differs")
    if (previous / "task/functions.py").read_bytes() != case.source.read_bytes():
        raise ValueError("Existing functions.py differs from the benchmark")
    task = prepare_task(case, directory, device, steps, seed)
    result = finish_trial(
        case,
        directory,
        task,
        (previous / "training/checkpoints/final.ckpt").resolve(),
        device,
        steps,
        seed,
        float(original["training_seconds"]),
        smoke,
    )
    result["training_reused_from"] = str(previous.resolve())
    (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
