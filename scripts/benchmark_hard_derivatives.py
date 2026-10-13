"""CUDA product-rule screening on fixed weights, not a training-quality benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinnstorch.api import Coordinates, Equation  # noqa: E402
from pinnstorch.runner.artifacts import ArtifactWriter  # noqa: E402
from pinnstorch.runner.execution import SolverExecution  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402
from pinnstorch.training.constraints import HardBoxInitial  # noqa: E402
from pinnstorch.training.hard_derivatives import HardBoxDerivatives  # noqa: E402


def measure(operation: Callable[[], torch.Tensor]) -> dict[str, float | int]:
    for _ in range(10):
        operation()
    durations: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(30):
            operation()
        torch.cuda.synchronize()
        durations.append((time.perf_counter() - started) / 30)
    return {
        "median_seconds": statistics.median(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--conductivity", type=float, default=0.1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this experiment on the remote CUDA machine")
    if args.report.exists():
        raise ValueError("Refusing to overwrite an existing report")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = args.trial / "task"
    checkpoint = (args.trial / "training/checkpoints/final.ckpt").resolve()
    settings = TaskSettings.load(directory / "config.yaml")
    problem = load_problem(directory / "functions.py")
    if tuple(problem.fields) != ("u",) or len(problem.equations) != 1:
        raise ValueError("Screening requires a scalar heat problem")
    objects = SolverFactory(settings, problem, directory).build(True)
    SolverExecution(
        settings,
        objects,
        ArtifactWriter(directory, settings, problem),
        logging.getLogger(__name__),
    )._load(checkpoint, False)
    model = objects.model.to("cuda").train()
    transform = model.output_fn
    if not isinstance(transform, HardBoxInitial):
        raise ValueError("Checkpoint must use HardBoxInitial")
    generator = torch.Generator(device="cuda").manual_seed(20261013)
    rows: list[dict[str, object]] = []
    for batch_size in (128, 1024, 2048, 17):
        base = [
            torch.rand(batch_size, 1, device="cuda", generator=generator)
            .mul(right - left)
            .add(left)
            for left, right in [*transform.bounds, transform.time_interval]
        ]

        def operation(product_rule: bool) -> torch.Tensor:
            model.zero_grad(set_to_none=True)
            columns = [column.detach().requires_grad_() for column in base]
            coordinates = Coordinates(dict(zip(settings.domain.space, columns[:-1])), columns[-1])
            network = model.net(columns[:-1], columns[-1])
            if product_rule:
                residual = HardBoxDerivatives(transform, network, coordinates).heat_residual(
                    "u", args.conductivity
                )
            else:
                equation = Equation(transform(network, *columns), coordinates)
                residual = next(iter(problem.equations.values()))(equation)
            residual.square().mean().backward()
            return residual.detach()

        reference = operation(False)
        reference_gradient = torch.cat(
            [cast(torch.Tensor, parameter.grad).flatten() for parameter in model.parameters()]
        ).clone()
        actual = operation(True)
        actual_gradient = torch.cat(
            [cast(torch.Tensor, parameter.grad).flatten() for parameter in model.parameters()]
        )
        torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=2e-4, atol=2e-5)
        row: dict[str, object] = {
            "batch_size": batch_size,
            "max_residual_difference": float((actual - reference).abs().max()),
            "max_gradient_difference": float((actual_gradient - reference_gradient).abs().max()),
        }
        for repetition in range(3):
            for enabled in [False, True] if repetition % 2 == 0 else [True, False]:
                row[f"{'product_rule' if enabled else 'eager'}_{repetition}"] = measure(
                    lambda: operation(enabled)
                )
        rows.append(row)
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": "float32",
        "tf32": False,
        "conductivity": args.conductivity,
        "holdout_seed": 20261013,
        "rows": rows,
        "limitations": "Fixed-weight residual+backward only; no optimizer updates or time-to-quality",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
