"""Compile face values and check sampled IC/edge compatibility before training."""

from functools import partial
from itertools import combinations
from typing import cast

import numpy as np
import torch

from pinnstorch.contracts import TensorOutputs
from pinnstorch.training.dirichlet_lifting import DirichletLifting, FaceTrace

from .problem import ProblemAdapter
from .settings import Boundary, TaskSettings


def trace_value(
    boundary: Boundary, adapter: ProblemAdapter, spatial: list[torch.Tensor], time: torch.Tensor
) -> torch.Tensor:
    value = (
        adapter.boundary_value(boundary.value_function, spatial, time)
        if boundary.value_function is not None
        else torch.full_like(time, cast(float, boundary.value))
    )
    if value.ndim == 0:
        value = value.expand_as(time)
    if value.shape != time.shape:
        raise ValueError("Dirichlet trace must be a finite scalar or Tensor column")
    return value


def compile_lifting(settings: TaskSettings, adapter: ProblemAdapter) -> DirichletLifting | None:
    if all(face.value == 0 and face.value_function is None for face in settings.boundaries):
        return None
    axes = tuple(settings.domain.space)
    traces = tuple(
        FaceTrace(
            boundary.field,
            axes.index(boundary.axis),
            side,
            partial(trace_value, boundary, adapter),
        )
        for boundary in settings.boundaries
        for side in (
            (0, 1) if boundary.side == "both" else (0 if boundary.side == "lower" else 1,)
        )
    )
    bounds = tuple(axis.interval for axis in settings.domain.space.values())
    lifting = DirichletLifting.build(bounds, tuple(adapter.problem.fields), traces)
    validate_traces(settings, adapter, lifting, traces)
    return lifting


def require_matching(left: torch.Tensor, right: torch.Tensor, context: str) -> None:
    if not (torch.isfinite(left).all() and torch.isfinite(right).all()) or not torch.allclose(
        left, right, atol=1e-6, rtol=1e-6
    ):
        raise ValueError(f"Incompatible boundary data: {context}")


def validate_traces(
    settings: TaskSettings,
    adapter: ProblemAdapter,
    lifting: DirichletLifting,
    traces: tuple[FaceTrace, ...],
) -> None:
    grid = settings.domain.grid()
    spatial = [torch.as_tensor(column, dtype=torch.float64) for column in grid.spatial]
    initial_time = torch.full_like(spatial[0], settings.domain.time.interval[0])
    with torch.no_grad():
        for face in traces:
            projected = lifting.project(spatial, face.axis, face.side)
            initial: TensorOutputs = adapter.initial_tensors(projected, initial_time)
            require_matching(
                initial[face.field], face.value(projected, initial_time), "initial/face"
            )
        for instant in np.asarray(grid.time).reshape(-1):
            time = torch.full_like(initial_time, float(instant))
            for left, right in combinations(traces, 2):
                if left.field != right.field or left.axis == right.axis:
                    continue
                projected = lifting.project(
                    lifting.project(spatial, left.axis, left.side), right.axis, right.side
                )
                require_matching(
                    left.value(projected, time), right.value(projected, time), "edge/corner traces"
                )
