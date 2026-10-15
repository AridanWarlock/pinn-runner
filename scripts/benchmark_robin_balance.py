"""Check physical boundary operators and integral heat balance of trained 1D cases."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.box_heat_balance import box_heat_balance  # noqa: E402
from benchmarks.cases import CASES  # noqa: E402
from benchmarks.heat_balance import heat_balance  # noqa: E402
from pinnstorch.runner.artifacts import ArtifactWriter  # noqa: E402
from pinnstorch.runner.dirichlet import trace_value  # noqa: E402
from pinnstorch.runner.execution import SolverExecution  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402

LIMITS = {
    "relative_l2": 0.01,
    "final_relative_l2": 0.02,
    "pde_rmse": 0.02,
    "boundary_operator_rmse": 1e-4,
    "heat_balance_rate_rmse": 0.01,
    "heat_balance_accumulated_max_error": 0.005,
    "initial_rmse": 1e-5,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--space-points", type=int)
    parser.add_argument("--time-points", type=int)
    parser.add_argument(
        "--quadrature", choices=("trapezoid", "gauss_legendre"), default="trapezoid"
    )
    args = parser.parse_args()
    if any(value is not None and value < 2 for value in (args.space_points, args.time_points)):
        parser.error("Quadrature counts must exceed one")
    if not torch.cuda.is_available():
        raise RuntimeError("This numerical benchmark requires remote CUDA")
    if args.report.exists():
        raise ValueError("Report already exists")
    result = json.loads((args.trial / "result.json").read_text())
    name = result["case"]
    if name not in (
        "heat_neumann",
        "heat_robin",
        "heat_neumann2d",
        "heat_neumann3d",
        "heat_robin2d",
        "heat_robin3d",
        "heat_mixed3d",
    ):
        raise ValueError("Balance benchmark only supports the documented Robin/Neumann cases")
    task = args.trial / "task"
    if (task / "functions.py").read_bytes() != CASES[name].source.read_bytes():
        raise ValueError(
            "Benchmark source differs; cannot assume the documented heat equation/source"
        )
    settings = TaskSettings.load(task / "config.yaml")
    settings = replace(
        settings,
        training=replace(
            settings.training,
            hard_execution=result.get("hard_execution", "eager"),
            hard_lifting=result.get("hard_lifting", "recursive"),
        ),
    )
    problem = load_problem(task / "functions.py")
    factory = SolverFactory(settings, problem, args.trial, task)
    objects = factory.build(predicting=True)
    SolverExecution(
        settings,
        objects,
        ArtifactWriter(args.trial, settings, problem),
        logging.getLogger(__name__),
    )._load(args.trial / "training/checkpoints/final.ckpt", False)
    model = objects.model.to("cuda").eval()

    def field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model.forward([x], t)["u"]

    def target(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return factory.adapter.boundary_value("wall", [x], t)

    def source(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x

    space_points = args.space_points or (257 if CASES[name].dimensions == 1 else 17)
    time_points = args.time_points or (31 if CASES[name].dimensions == 1 else 21)
    torch.cuda.synchronize()
    started = time.perf_counter()
    if CASES[name].dimensions == 1 and args.quadrature == "trapezoid":
        metrics = heat_balance(
            field,
            settings.domain.space["x"].interval,
            settings.domain.time.interval,
            "cuda",
            math.pi / 2 if name == "heat_robin" else 0,
            1,
            target,
            source,
            space_points=space_points,
            time_points=time_points,
        )
    else:

        def box_field(spatial: list[torch.Tensor], clock: torch.Tensor) -> torch.Tensor:
            return model.forward(spatial, clock)["u"]

        def box_source(spatial: list[torch.Tensor], clock: torch.Tensor) -> torch.Tensor:
            value = torch.zeros_like(clock)
            for column in spatial:
                value = value + column
            return value

        names = tuple(settings.domain.space)
        lookup = {boundary.axis: boundary for boundary in settings.boundaries}

        from functools import partial

        targets = tuple(
            (partial(trace_value, lookup[axis], factory.adapter),) * 2 for axis in names
        )
        operators = tuple(
            ((lookup[axis].alpha if lookup[axis].kind == "robin" else 0, 1),) * 2 for axis in names
        )
        metrics = box_heat_balance(
            box_field,
            targets,
            box_source,
            tuple(axis.interval for axis in settings.domain.space.values()),
            settings.domain.time.interval,
            operators,
            "cuda",
            space_points=space_points,
            time_points=time_points,
            quadrature=args.quadrature,
        )
    torch.cuda.synchronize()
    diagnostic_seconds = time.perf_counter() - started
    limits = dict(LIMITS)
    if CASES[name].dimensions > 1:
        limits["mode_relative_l2"] = 0.02
    combined = {**result["metrics"], **metrics}
    failures = [
        f"{key}={combined[key]:.6g} exceeds {limit}"
        for key, limit in limits.items()
        if not math.isfinite(combined[key]) or combined[key] > limit
    ]
    report = {
        "case": name,
        "seed": result["seed"],
        "constraint_kind": result["constraint_kind"],
        "evaluation_execution": settings.training.hard_execution,
        "evaluation_lifting": settings.training.hard_lifting,
        "limits": limits,
        "metrics": combined,
        "status": "failed" if failures else "passed",
        "failures": failures,
        "space_points": space_points,
        "quadrature": args.quadrature,
        "time_points": time_points,
        "conductivity": 0.1,
        "source": "sum of spatial coordinates",
        "diagnostic_seconds": diagnostic_seconds,
        "limitations": "Quadrature error included; physical autograd on a grid independent of training points",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
