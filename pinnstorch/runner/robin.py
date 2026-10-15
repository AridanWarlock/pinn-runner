"""Resolve one-field interval boundary operators and validate initial traces."""

from functools import partial

import torch

from pinnstorch.api import Problem
from pinnstorch.training.robin_batch import BatchedBoxRobinProjection
from pinnstorch.training.robin_box import BoxRobinInitial, BoxRobinProjection
from pinnstorch.training.robin_grouped import GroupedBoxRobinProjection
from pinnstorch.training.robin_initial import IntervalRobinInitial
from pinnstorch.training.robin_lifting import ExpandedRobinLifting
from pinnstorch.training.robin_projection import IntervalRobinProjection, RobinOperator

from .dirichlet import require_matching, trace_value
from .problem import ProblemAdapter
from .robin_validation import validate_box_robin
from .settings import Boundary, TaskSettings


def compile_robin(
    settings: TaskSettings, problem: Problem, adapter: ProblemAdapter
) -> IntervalRobinInitial | BoxRobinInitial:
    if settings.domain.parameters or len(problem.fields) != 1:
        raise ValueError("hard_initial_robin requires one field and no parameters")
    if problem.transform is not None or tuple(problem.outputs) != tuple(problem.fields):
        raise ValueError("hard_initial_robin does not support custom transforms/network fields")
    if (
        len(settings.domain.space) > 1
        or settings.training.hard_execution != "eager"
        or settings.training.hard_lifting != "recursive"
    ):
        return compile_box(settings, adapter)
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


def compile_box(settings: TaskSettings, adapter: ProblemAdapter) -> BoxRobinInitial:
    names = tuple(settings.domain.space)
    faces: dict[tuple[str, str], Boundary] = {}
    for boundary in settings.boundaries:
        if boundary.kind not in ("neumann", "robin") or boundary.axis not in names:
            raise ValueError("hard_initial_robin requires Neumann/Robin on every face")
        for side in ("lower", "upper") if boundary.side == "both" else (boundary.side,):
            key = (boundary.axis, side)
            if key in faces:
                raise ValueError("hard_initial_robin requires one condition per face")
            faces[key] = boundary
    if set(faces) != {(name, side) for name in names for side in ("lower", "upper")}:
        raise ValueError("hard_initial_robin requires all box faces")
    projections = []
    targets = []
    for name in names:
        pair = (faces[name, "lower"], faces[name, "upper"])
        operators = tuple(
            RobinOperator(0, 1) if b.kind == "neumann" else RobinOperator(b.alpha, b.beta)
            for b in pair
        )
        projections.append(
            IntervalRobinProjection(settings.domain.space[name].interval, *operators)
        )
        targets.append(
            (partial(trace_value, pair[0], adapter), partial(trace_value, pair[1], adapter))
        )
    execution_types = {
        "eager": BoxRobinProjection,
        "robin_batch": BatchedBoxRobinProjection,
        "robin_grouped": GroupedBoxRobinProjection,
    }
    projection_type = execution_types[settings.training.hard_execution]
    recipe = BoxRobinInitial(
        projection_type(tuple(projections)),
        settings.domain.time.interval,
        adapter.initial_tensors,
        tuple(targets),
        adapter.problem.fields[0],
        lifting=(
            ExpandedRobinLifting(tuple(projections), tuple(targets))
            if settings.training.hard_lifting == "expanded"
            else None
        ),
    )
    validate_box_robin(settings, recipe)
    return recipe
