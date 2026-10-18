"""Diagnostic allocator observations, separated from unprofiled timing."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from pinnstorch.models.pinn_module import PINNModule
from pinnstorch.runner.factory import SolverFactory

STAGES = (
    "network_and_features",
    "background_reconstruction",
    "physical_residual",
    "loss_reduction",
    "parameter_gradient",
)


def reduce_memory_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    events = trace["traceEvents"]
    ranges = [e for e in events if e.get("ph") == "X" and e.get("name") in STAGES]
    if [e["name"] for e in ranges] != list(STAGES) or any(
        a["ts"] + a["dur"] > b["ts"] for a, b in zip(ranges, ranges[1:])
    ):
        raise ValueError("Expected five nonoverlapping allocator regions")
    rows = []
    for region in ranges:
        samples = [
            e
            for e in events
            if e.get("name") == "[memory]"
            and e.get("pid") == region["pid"]
            and region["ts"] <= e["ts"] < region["ts"] + region["dur"]
            and e.get("args", {}).get("Device Type") == 1
        ]
        sizes = [e["args"]["Bytes"] for e in samples]
        rows.append(
            dict(
                stage=region["name"],
                allocation_events=sum(v > 0 for v in sizes),
                free_events=sum(v < 0 for v in sizes),
                allocated_bytes=sum(v for v in sizes if v > 0),
                freed_bytes=-sum(v for v in sizes if v < 0),
            )
        )
    return rows


def profile_allocations(
    factory: SolverFactory, model: PINNModule, points: torch.Tensor, output: Path
) -> dict[str, Any]:
    columns = [
        points[:, i : i + 1].detach().clone().requires_grad_() for i in range(points.shape[1])
    ]
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    rows: list[dict[str, Any]] = []

    def measure(stage: str, operation: Callable[[], Any]) -> Any:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        with torch.profiler.record_function(stage):
            value = operation()
            torch.cuda.synchronize()
        rows.append(
            dict(
                stage=stage,
                before_bytes=before,
                after_bytes=torch.cuda.memory_allocated(),
                peak_bytes=torch.cuda.max_memory_allocated(),
                reserved_before_bytes=reserved_before,
                reserved_after_bytes=torch.cuda.memory_reserved(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        )
        return value

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
    ) as profile:
        raw = measure(STAGES[0], lambda: model.net(columns[:-1], columns[-1]))
        if model.output_fn is None:
            raise ValueError("Allocation profile requires the existing hard output transform")
        transform = model.output_fn
        physical = measure(STAGES[1], lambda: dict(transform(raw, *columns)))
        residuals = measure(STAGES[2], lambda: factory.adapter.residuals(physical, *columns))
        loss = measure(
            STAGES[3],
            lambda: torch.stack(
                [residuals[n].square().mean() for n in factory.problem.equations]
            ).sum(),
        )
        measure(STAGES[4], lambda: torch.autograd.backward(loss, inputs=tuple(model.parameters())))
    path = output / "allocator-trace.json"
    profile.export_chrome_trace(str(path))
    allocations = reduce_memory_events(json.loads(path.read_text()))
    for row, allocation in zip(rows, allocations, strict=True):
        row.update(allocation)
    return dict(
        stages=rows,
        loss_value=float(loss.detach()),
        cuda_memory_events=sum(r["allocation_events"] + r["free_events"] for r in rows),
        cuda_kernel_timing=False,
        profiled=True,
        performance_claim=False,
    )
