"""Validated mathematical guarantees and remaining objective, independent of execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pinnstorch.api import Problem
from pinnstorch.contracts import FieldFunction
from pinnstorch.training.constraints import HardBoxInitial
from pinnstorch.training.hard_execution import hard_executor

from .problem import ProblemAdapter
from .settings import TaskSettings


@dataclass(frozen=True)
class ExactFace:
    field: str
    axis: str
    side: str
    value: float = 0.0


@dataclass(frozen=True)
class ConstraintPlan:
    kind: str
    exact_initial: tuple[str, ...]
    exact_faces: tuple[ExactFace, ...]
    remaining_losses: tuple[str, ...]
    signature: str
    recipe: HardBoxInitial | None
    output: FieldFunction | None

    @classmethod
    def resolve(
        cls, settings: TaskSettings, problem: Problem, adapter: ProblemAdapter, directory: Path
    ) -> ConstraintPlan:
        if settings.training.constraints == "soft":
            losses = ("initial", "equations") + (("boundaries",) if settings.boundaries else ())
            return cls(
                "soft", (), (), losses, "", None, adapter.transform if problem.transform else None
            )
        validate_box_initial(settings, problem, adapter)
        recipe = HardBoxInitial(
            tuple(axis.interval for axis in settings.domain.space.values()),
            settings.domain.time.interval,
            adapter.initial_tensors,
            spatial_names=tuple(settings.domain.space),
        )
        faces = tuple(
            ExactFace(field, axis, side)
            for field in problem.fields
            for axis in settings.domain.space
            for side in ("lower", "upper")
        )
        signature = hashlib.sha256((directory / "functions.py").read_bytes()).hexdigest()
        return cls(
            settings.training.constraints,
            tuple(problem.fields),
            faces,
            ("equations",),
            signature,
            recipe,
            hard_executor(recipe, settings.training.hard_execution),
        )


def validate_box_initial(
    settings: TaskSettings, problem: Problem, adapter: ProblemAdapter
) -> None:
    if settings.training.constraints == "soft":
        return
    if problem.transform is not None or tuple(problem.outputs) != tuple(problem.fields):
        raise ValueError("Hard constraints do not support custom transforms/network fields")
    if settings.domain.parameters:
        raise ValueError("Hard constraints with variable parameters are not yet supported")
    required = {
        (field, axis, side)
        for field in problem.fields
        for axis in settings.domain.space
        for side in ("lower", "upper")
    }
    provided: set[tuple[str, str, str]] = set()
    for boundary in settings.boundaries:
        if (
            boundary.kind != "dirichlet"
            or boundary.value != 0
            or boundary.value_function is not None
        ):
            raise ValueError("Hard constraints require homogeneous constant Dirichlet faces")
        sides = ("lower", "upper") if boundary.side == "both" else (boundary.side,)
        provided.update((boundary.field, boundary.axis, side) for side in sides)
    if provided != required:
        raise ValueError("Hard constraints require all faces for every physical field")
    validate_initial_trace(settings, adapter)


def validate_initial_trace(settings: TaskSettings, adapter: ProblemAdapter) -> None:
    grid = settings.domain.grid()
    mask = np.zeros(len(grid.spatial[0]), dtype=bool)
    for index, axis in enumerate(settings.domain.space.values()):
        column = grid.spatial[index][:, 0]
        mask |= np.isclose(column, axis.interval[0]) | np.isclose(column, axis.interval[1])
    values = adapter.initial(np.column_stack([column[:, 0] for column in grid.spatial]))
    if any(not np.allclose(value[mask], 0, atol=1e-6, rtol=0) for value in values.values()):
        raise ValueError("Initial values incompatible with homogeneous Dirichlet faces")
