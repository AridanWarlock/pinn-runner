"""Resolve one-field interval boundary operators and validate initial traces."""

from functools import partial

import torch

from pinnstorch.api import Problem
from pinnstorch.training.robin_initial import IntervalRobinInitial
from pinnstorch.training.robin_projection import IntervalRobinProjection, RobinOperator

from .dirichlet import require_matching, trace_value
from .problem import ProblemAdapter
from .settings import Boundary, TaskSettings


def compile_robin(
    settings: TaskSettings, problem: Problem, adapter: ProblemAdapter
) -> IntervalRobinInitial:
    if len(settings.domain.space) != 1 or settings.domain.parameters or len(problem.fields) != 1:
        raise ValueError(
            "hard_initial_robin requires one spatial axis, one field and no parameters"
        )
    if problem.transform is not None or tuple(problem.outputs) != tuple(problem.fields):
        raise ValueError("hard_initial_robin does not support custom transforms/network fields")
    axis = next(iter(settings.domain.space))
    faces: dict[str, Boundary] = {}
    for boundary in settings.boundaries:
        if boundary.kind not in ("neumann", "robin") or boundary.axis != axis:
            raise ValueError("hard_initial_robin requires Neumann/Robin on both endpoints")
        for side in ("lower", "upper") if boundary.side == "both" else (boundary.side,):
            if side in faces:
                raise ValueError("hard_initial_robin requires one condition per endpoint")
            faces[side] = boundary
    if set(faces) != {"lower", "upper"}:
        raise ValueError("hard_initial_robin requires both endpoints")
    operators = tuple(
        RobinOperator(0, 1) if face.kind == "neumann" else RobinOperator(face.alpha, face.beta)
        for face in (faces["lower"], faces["upper"])
    )
    projection = IntervalRobinProjection(
        settings.domain.space[axis].interval, operators[0], operators[1]
    )
    targets = (
        partial(trace_value, faces["lower"], adapter),
        partial(trace_value, faces["upper"], adapter),
    )
    recipe = IntervalRobinInitial(
        projection,
        settings.domain.time.interval,
        adapter.initial_tensors,
        targets,
        problem.fields[0],
    )
    time = torch.tensor([[settings.domain.time.interval[0]]], dtype=torch.float64)
    x = torch.zeros_like(time)

    def initial(coordinate: torch.Tensor, clock: torch.Tensor) -> dict[str, torch.Tensor]:
        return adapter.initial_tensors([coordinate], clock)

    with torch.set_grad_enabled(True):
        actual = projection.traces(initial, x, time)
        for face_index, target in enumerate(targets):
            face = torch.full_like(x, projection.bounds[face_index])
            require_matching(
                actual[face_index][recipe.field],
                target([face], time),
                "initial Robin/Neumann operator",
            )
    return recipe
