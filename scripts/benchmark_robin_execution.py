"""Compare exact Robin execution adapters on the same saved CUDA physical model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.performance import prediction_speed  # noqa: E402
from pinnstorch.contracts import TensorOutputs  # noqa: E402
from pinnstorch.models.pinn_module import PINNModule  # noqa: E402
from pinnstorch.runner.artifacts import ArtifactWriter  # noqa: E402
from pinnstorch.runner.execution import SolverExecution  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402


def load_model(
    trial: Path, backend: str, lifting: str | None = None, device: str = "cuda"
) -> tuple[SolverFactory, PINNModule]:
    task = trial / "task"
    settings = TaskSettings.load(task / "config.yaml")
    if settings.training.constraints != "hard_initial_robin":
        raise ValueError("Execution comparison requires an exact Robin checkpoint")
    settings = replace(
        settings,
        runtime=replace(settings.runtime, device=device, devices=1),
        training=replace(
            settings.training,
            hard_execution=backend,
            hard_lifting=lifting if lifting is not None else settings.training.hard_lifting,
        ),
    )
    problem = load_problem(task / "functions.py")
    factory = SolverFactory(settings, problem, trial, task)
    objects = factory.build(True)
    SolverExecution(
        settings, objects, ArtifactWriter(trial, settings, problem), logging.getLogger(__name__)
    )._load(trial / "training/checkpoints/final.ckpt", False)
    return factory, objects.model.to(device).eval()


def point_columns(points: torch.Tensor) -> list[torch.Tensor]:
    return [points[:, index : index + 1] for index in range(points.shape[1])]


def objective(factory: SolverFactory, model: PINNModule, points: torch.Tensor) -> torch.Tensor:
    columns = [x.detach().requires_grad_() for x in point_columns(points)]
    outputs = dict(model(columns[:-1], columns[-1]))
    residuals = factory.adapter.residuals(outputs, *columns)
    return equation_loss(factory, residuals)


def equation_loss(factory: SolverFactory, outputs: TensorOutputs) -> torch.Tensor:
    # The adapter returns physical fields alongside named equation residuals.
    values = [outputs[name].square().mean() for name in factory.problem.equations]
    return torch.sum(torch.stack(values))


def training_speed(
    factory: SolverFactory, model: PINNModule, points: torch.Tensor
) -> dict[str, float | int]:
    durations = []
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    for iteration in range(12):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        torch.autograd.backward(objective(factory, model, points))
        torch.cuda.synchronize()
        if iteration >= 3:
            durations.append(time.perf_counter() - started)
    peak = torch.cuda.max_memory_allocated() - baseline
    model.zero_grad(set_to_none=True)
    return {
        "training_objective_seconds_median": statistics.median(durations),
        "training_objective_seconds_min": min(durations),
        "training_objective_seconds_max": max(durations),
        "incremental_peak_allocated_bytes": peak,
        "measurement_passes": 9,
        "warmup_passes": 3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Execution comparison requires remote CUDA")
    if args.report.exists():
        raise ValueError("Refusing to overwrite an existing report")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = {
        backend: load_model(args.trial, backend)
        for backend in ("eager", "robin_batch", "robin_grouped")
    }
    settings = models["eager"][0].settings
    generator = torch.Generator(device="cuda").manual_seed(20261014)
    points = torch.rand((512, len(settings.domain.space) + 1), device="cuda", generator=generator)
    for column, axis in zip(
        point_columns(points), (*settings.domain.space.values(), settings.domain.time)
    ):
        column.mul_(axis.interval[1] - axis.interval[0]).add_(axis.interval[0])
    values, gradients, losses = {}, {}, {}
    for backend, (factory, model) in models.items():
        model.zero_grad(set_to_none=True)
        columns = list(point_columns(points))
        with torch.no_grad():
            values[backend] = model(columns[:-1], columns[-1])["u"]
        loss = objective(factory, model, points)
        losses[backend] = float(loss.detach().cpu())
        torch.autograd.backward(loss)
        gradients[backend] = torch.cat(
            [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
        )
        model.zero_grad(set_to_none=True)
    equivalence = {}
    for backend in ("robin_batch", "robin_grouped"):
        torch.testing.assert_close(values[backend], values["eager"], atol=3e-5, rtol=3e-4)
        torch.testing.assert_close(gradients[backend], gradients["eager"], atol=3e-5, rtol=3e-4)
        equivalence[backend] = dict(
            value_max_difference=float((values[backend] - values["eager"]).abs().max().cpu()),
            gradient_max_difference=float(
                (gradients[backend] - gradients["eager"]).abs().max().cpu()
            ),
        )
    rows: list[dict[str, Any]] = []
    for repetition in range(3):
        for size in (128, 512):
            order = ("eager", "robin_batch", "robin_grouped")
            for backend in order if repetition % 2 == 0 else tuple(reversed(order)):
                factory, model = models[backend]
                selected = points[:size]
                rows.append(
                    {
                        "repeat": repetition,
                        "backend": backend,
                        "batch_size": size,
                        **training_speed(factory, model, selected),
                        **prediction_speed(
                            model, point_columns(selected[:, :-1]), selected[:, -1:], size
                        ),
                    }
                )
    report = {
        "trial": str(args.trial),
        "checkpoint_sha256": hashlib.sha256(
            (args.trial / "training/checkpoints/final.ckpt").read_bytes()
        ).hexdigest(),
        "objective_values": losses,
        "implementation_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "gpu": torch.cuda.get_device_name(0),
        "equivalence": equivalence,
        "rows": rows,
        "limitations": "Same fixed parameters; measures objective/backprop without optimizer updates. Not a convergence or time-to-quality result.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
