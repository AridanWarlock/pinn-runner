"""Short fixed-checkpoint CUDA comparison of two-pass and shared-geometry lifting."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402
from pinnstorch.training.dirichlet_lifting import DirichletLifting  # noqa: E402
from pinnstorch.training.hard_execution import hard_executor, hard_recipe  # noqa: E402
from scripts.benchmark_solutions import environment  # noqa: E402


def timing(operation: Callable[[], torch.Tensor]) -> dict[str, float | int]:
    for _ in range(3):
        operation()
    durations = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(10):
            operation()
        torch.cuda.synchronize()
        durations.append((time.perf_counter() - started) / 10)
    return {
        "seconds": sorted(durations)[2],
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    if args.report.exists():
        raise ValueError("Report already exists")
    task = args.trial / "task"
    settings = TaskSettings.load(task / "config.yaml")
    problem = load_problem(task / "functions.py")
    factory = SolverFactory(settings, problem, args.trial, task)
    model = factory.build(predicting=True).model.to("cuda").eval()
    checkpoint = args.trial / "training/checkpoints/final.ckpt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"], strict=True)
    recipe = hard_recipe(model.output_fn)
    if not isinstance(recipe.lifting, DirichletLifting):
        raise ValueError("Requires compiled nonzero Dirichlet lifting")
    original = replace(recipe, lifting=recipe.lifting.__call__)
    rows: list[dict[str, object]] = []
    generator = torch.Generator(device="cuda").manual_seed(20261013)
    for count in (128, 2048, 17):
        base = [
            torch.rand(count, 1, generator=generator, device="cuda").mul(right - left).add(left)
            for left, right in (*recipe.bounds, recipe.time_interval)
        ]
        for training in (True, False):

            def operation() -> torch.Tensor:
                model.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    columns = [column.detach().requires_grad_(training) for column in base]
                    outputs = model(columns[:-1], columns[-1])
                    if not training:
                        return cast(torch.Tensor, outputs["u"].detach())
                    values = factory.adapter.residuals(outputs, *columns)
                    residual = torch.cat([values[name] for name in problem.equations], dim=1)
                    residual.square().mean().backward()
                    return residual.detach()

            model.output_fn = hard_executor(original, "product_rule")
            reference = operation().clone()
            reference_gradient = (
                torch.cat(
                    [cast(torch.Tensor, p.grad).flatten() for p in model.parameters()]
                ).clone()
                if training
                else None
            )
            model.output_fn = hard_executor(recipe, "product_rule")
            actual = operation()
            torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-5)
            row: dict[str, object] = {
                "batch_size": count,
                "training": training,
                "max_difference": float((actual - reference).abs().max()),
            }
            if reference_gradient is not None:
                gradient = torch.cat(
                    [cast(torch.Tensor, p.grad).flatten() for p in model.parameters()]
                )
                torch.testing.assert_close(gradient, reference_gradient, rtol=2e-4, atol=2e-5)
                row["max_gradient_difference"] = float((gradient - reference_gradient).abs().max())
            for repetition in range(2):
                for enabled in (False, True) if repetition == 0 else (True, False):
                    model.output_fn = hard_executor(
                        recipe if enabled else original, "product_rule"
                    )
                    row[f"{'shared' if enabled else 'two_pass'}_{repetition}"] = timing(operation)
            rows.append(row)
    report = {
        "environment": environment("cuda"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "rows": rows,
        "warmup": 3,
        "samples": 5,
        "calls_per_sample": 10,
        "limitations": "Fixed weights, same CUDA points; no optimizer updates or cold end-to-end timing",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("passed; " + str(args.report))


if __name__ == "__main__":
    main()
