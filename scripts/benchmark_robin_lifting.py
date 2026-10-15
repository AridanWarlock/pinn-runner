"""Separate Robin data-lifting cost from network projection on fixed saved weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinnstorch.api import Coordinates, Equation  # noqa: E402
from pinnstorch.models.pinn_module import PINNModule  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.training.robin_box import BoundBoxRobinInitial  # noqa: E402
from scripts.benchmark_robin_execution import load_model, objective, point_columns  # noqa: E402


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def measure(action: Callable[[], None], device: str) -> dict[str, float | int]:
    times = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated() if device == "cuda" else 0
    for iteration in range(12):
        synchronize(device)
        started = time.perf_counter()
        action()
        synchronize(device)
        if iteration >= 3:
            times.append(time.perf_counter() - started)
    return {
        "seconds_median": statistics.median(times),
        "seconds_min": min(times),
        "seconds_max": max(times),
        "incremental_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() - baseline if device == "cuda" else 0
        ),
    }


def actions(
    factory: SolverFactory, model: PINNModule, points: torch.Tensor
) -> dict[str, Callable[[], None]]:
    bound = model.output_fn
    if not isinstance(bound, BoundBoxRobinInitial):
        raise ValueError("Requires a box Robin model")
    names = tuple(factory.settings.domain.space)
    lower, upper = factory.settings.domain.time.interval

    def component(kind: str) -> None:
        model.zero_grad(set_to_none=True)
        columns = [x.detach().requires_grad_() for x in point_columns(points)]
        spatial, clock = columns[:-1], columns[-1]
        if kind == "full_objective_backward":
            torch.autograd.backward(objective(factory, model, points))
            model.zero_grad(set_to_none=True)
            return
        if kind == "prediction":
            with torch.no_grad():
                model(spatial, clock)
            return
        if kind.startswith("background"):
            outputs = bound.recipe.background(spatial, clock)
            if kind == "background_forward":
                return
        else:
            raw = bound.network(spatial, clock)
            correction = bound.recipe.projection.apply(raw, bound.network, spatial, clock)
            outputs = {
                name: value * (clock - lower) / (upper - lower)
                for name, value in correction.items()
            }
        equation = Equation(outputs, Coordinates(dict(zip(names, spatial)), clock))
        residual = cast(torch.Tensor, equation.dt("u") - 0.1 * equation.laplacian("u"))
        if kind == "network_objective_backward":
            torch.autograd.backward(residual.square().mean())
            model.zero_grad(set_to_none=True)

    from functools import partial

    return {
        name: partial(component, name)
        for name in (
            "background_forward",
            "background_physical_derivatives",
            "network_objective_backward",
            "full_objective_backward",
            "prediction",
        )
    }


def signature(
    factory: SolverFactory, model: PINNModule, points: torch.Tensor
) -> dict[str, torch.Tensor]:
    columns = [x.detach().requires_grad_() for x in point_columns(points)]
    outputs = dict(model(columns[:-1], columns[-1]))
    residuals = factory.adapter.residuals(outputs, *columns)
    residual = torch.cat(tuple(residuals.values()), dim=1)
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(residual.square().mean(), parameters, allow_unused=True)
    return {
        "field": outputs["u"].detach(),
        "residual": residual.detach(),
        "parameter_gradient": torch.cat(
            [
                (torch.zeros_like(parameter) if gradient is None else gradient).detach().flatten()
                for parameter, gradient in zip(parameters, gradients)
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--trace", action="store_true", help="Separate diagnostic profiler pass")
    args = parser.parse_args()
    if args.report.exists() or min(args.batch_sizes) < 1:
        parser.error("New report path and positive batch sizes required")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; no fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    models = {
        kind: load_model(args.trial, "robin_grouped", kind, args.device)
        for kind in ("recursive", "expanded")
    }
    settings = models["recursive"][0].settings
    generator = torch.Generator(device=args.device).manual_seed(20261015)
    points = torch.rand(
        (max(args.batch_sizes), len(settings.domain.space) + 1),
        device=args.device,
        generator=generator,
    )
    for column, axis in zip(
        point_columns(points), (*settings.domain.space.values(), settings.domain.time)
    ):
        column.mul_(axis.interval[1] - axis.interval[0]).add_(axis.interval[0])
    control = signature(*models["recursive"], points)
    candidate = signature(*models["expanded"], points)
    differences = {}
    for name, expected in control.items():
        torch.testing.assert_close(candidate[name], expected, atol=3e-5, rtol=3e-4)
        differences[name] = float((candidate[name] - expected).abs().max().cpu())
    rows: list[dict[str, Any]] = []
    for repeat in range(3):
        for size in args.batch_sizes:
            for kind in (
                ("recursive", "expanded") if repeat % 2 == 0 else ("expanded", "recursive")
            ):
                for component, action in actions(*models[kind], points[:size]).items():
                    rows.append(
                        {
                            "repeat": repeat,
                            "lifting": kind,
                            "batch_size": size,
                            "component": component,
                            **measure(action, args.device),
                        }
                    )
    profile = []
    if args.trace:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        for kind, pair in models.items():
            action = actions(*pair, points)["full_objective_backward"]
            with torch.profiler.profile(
                activities=activities, record_shapes=True, profile_memory=True
            ) as profiler:
                action()
                synchronize(args.device)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(args.report.with_suffix(f".{kind}.trace.json")))
            profile.append(
                {
                    "lifting": kind,
                    "events": [
                        {
                            "name": event.key,
                            "count": event.count,
                            "self_cpu_time_us": event.self_cpu_time_total,
                            "self_cuda_time_us": event.self_cuda_time_total,
                            "self_cuda_memory_bytes": event.self_cuda_memory_usage,
                        }
                        for event in sorted(
                            profiler.key_averages(),
                            key=lambda event: event.self_cuda_time_total,
                            reverse=True,
                        )[:30]
                    ],
                }
            )
    report = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "checkpoint_sha256": hashlib.sha256(
            (args.trial / "training/checkpoints/final.ckpt").read_bytes()
        ).hexdigest(),
        "trial": str(args.trial),
        "device": args.device,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "equivalence_max_differences": differences,
        "rows": rows,
        "profile": profile,
        "limitations": "Fixed weights, physical autograd, no optimizer update. Components measured independently are not an additive wall-time decomposition. CPU runs are tool checks, not CUDA evidence. Profiler excluded from timing.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(args.report), "equivalence": differences}))


if __name__ == "__main__":
    main()
