"""Profile a saved convergence state without training or changing its weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.cases import CASES  # noqa: E402
from benchmarks.fingerprints import tensor_fingerprint  # noqa: E402
from benchmarks.network_work import count_network_work  # noqa: E402
from pinnstorch.models.pinn_module import PINNModule  # noqa: E402
from pinnstorch.runner.checkpoint_io import load_checkpoint  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402


def training_step(factory: SolverFactory, model: PINNModule, points: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    columns = [
        points[:, i : i + 1].detach().clone().requires_grad_() for i in range(points.shape[1])
    ]
    physical = dict(model(columns[:-1], columns[-1]))
    residuals = factory.adapter.residuals(physical, *columns)
    loss = cast(
        torch.Tensor,
        torch.stack([residuals[name].square().mean() for name in factory.problem.equations]).sum(),
    )
    torch.autograd.backward(loss, inputs=tuple(model.parameters()))
    return loss.detach()


def timed_steps(operation: Callable[[], torch.Tensor]) -> dict[str, float | int]:
    durations = []
    for index in range(12):
        torch.cuda.synchronize()
        started = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        if index >= 3:
            durations.append(time.perf_counter() - started)
    return {
        "median_seconds": statistics.median(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "warmup": 3,
        "measurements": len(durations),
    }


def operator_profile(operation: Callable[[], torch.Tensor], output: Path) -> dict[str, Any]:
    supported = cast(
        Callable[[], set[torch.profiler.ProfilerActivity]], torch.profiler.supported_activities
    )()
    activities = [torch.profiler.ProfilerActivity.CPU]
    cuda = torch.profiler.ProfilerActivity.CUDA in supported
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
        operation()
        torch.cuda.synchronize()
    rows = [
        {
            "operator": event.key,
            "count": event.count,
            "self_cpu_microseconds": event.self_cpu_time_total,
            "self_cuda_microseconds": event.self_cuda_time_total if cuda else None,
        }
        for event in profile.key_averages()
    ]
    rows.sort(key=lambda row: float(row["self_cpu_microseconds"]), reverse=True)
    with (output / "operators.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    profile.export_chrome_trace(str(output / "trace.json"))
    return {
        "cuda_activity_supported": cuda,
        "cuda_operator_time_recorded": any(
            (row["self_cuda_microseconds"] or 0) > 0 for row in rows
        ),
        "operators": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--weights", default="lb-150.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=1024)
    args = parser.parse_args()
    if args.points < 1 or args.output.exists():
        parser.error("Positive point count and a new output directory required")
    if not torch.cuda.is_available():
        parser.error("This profiler requires the remote Windows CUDA environment")
    settings = TaskSettings.load(args.trial / "config.yaml")
    if settings.runtime.device != "cuda":
        parser.error("Saved trial must use CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    case = CASES[args.case]
    factory = SolverFactory(settings, load_problem(case.source), args.trial, case.source.parent)
    model = factory.build(predicting=True).model.to("cuda").eval()
    weights = args.trial / args.weights
    model.load_state_dict(load_checkpoint(weights, map_location="cuda"))
    initial_fingerprint = tensor_fingerprint(model.state_dict())
    generator = torch.Generator(device="cpu").manual_seed(19)
    unit = torch.rand((args.points, len(settings.domain.space) + 1), generator=generator)
    intervals = [axis.interval for axis in settings.domain.space.values()]
    intervals.append(settings.domain.time.interval)
    points = torch.cat(
        [unit[:, i : i + 1] * (b - a) + a for i, (a, b) in enumerate(intervals)], dim=1
    ).to("cuda")

    def operation() -> torch.Tensor:
        return training_step(factory, model, points)

    work = count_network_work(model, operation)
    timing = timed_steps(operation)
    args.output.mkdir(parents=True)
    profiling = operator_profile(operation, args.output)
    model.zero_grad(set_to_none=True)
    if tensor_fingerprint(model.state_dict()) != initial_fingerprint:
        raise RuntimeError("Profiling changed the saved physical model")
    report = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "state_sha256": initial_fingerprint,
        "weights_unchanged": True,
        "case": args.case,
        "spatial_encoding": settings.network.spatial_encoding,
        "hard_execution": settings.training.hard_execution,
        "points": args.points,
        "network_work": asdict(work),
        "timing": timing,
        "profiler": profiling,
        "limitations": "One saved state, fixed independent points, no optimizer update. FCN hooks removed before timing. Profiling is separate from synchronized unprofiled timing. CUDA activity support does not guarantee that CUDA events were captured; inspect operators/trace before kernel conclusions. Raw trace stays outside Git.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
