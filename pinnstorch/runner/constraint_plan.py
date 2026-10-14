"""Validated mathematical guarantees and remaining objective, independent of execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pinnstorch.api import Problem
from pinnstorch.contracts import FieldFunction
from pinnstorch.training.constraints import HardBoxInitial
from pinnstorch.training.hard_execution import hard_executor
from pinnstorch.training.robin_initial import IntervalRobinInitial, NetworkField

from .dirichlet import compile_lifting
from .problem import ProblemAdapter
from .robin import compile_robin
from .settings import TaskSettings


@dataclass(frozen=True)
class ExactFace:
    field: str
    axis: str
    side: str
    value: float | None = 0.0
    value_function: str | None = None
    kind: str = "dirichlet"
    alpha: float = 1.0
    beta: float = 0.0


@dataclass(frozen=True)
class ConstraintPlan:
    kind: str
    exact_initial: tuple[str, ...]
    exact_faces: tuple[ExactFace, ...]
    remaining_losses: tuple[str, ...]
    signature: str
    recipe: HardBoxInitial | IntervalRobinInitial | None
    output: FieldFunction | None

    def bind(self, network: NetworkField) -> FieldFunction | None:
        if isinstance(self.recipe, IntervalRobinInitial):
            return self.recipe.bind(network)
        return self.output

    @classmethod
    def resolve(
        cls, settings: TaskSettings, problem: Problem, adapter: ProblemAdapter, directory: Path
    ) -> ConstraintPlan:
        if settings.training.constraints == "soft":
            losses = ("initial", "equations") + (("boundaries",) if settings.boundaries else ())
            return cls(
                "soft", (), (), losses, "", None, adapter.transform if problem.transform else None
            )
        if settings.training.constraints == "hard_initial_robin":
            return cls._robin(settings, problem, adapter, directory)
        validate_box_initial(settings, problem, adapter)
        lifting = compile_lifting(settings, adapter)
        recipe = HardBoxInitial(
            tuple(axis.interval for axis in settings.domain.space.values()),
            settings.domain.time.interval,
            adapter.initial_tensors,
            spatial_names=tuple(settings.domain.space),
            lifting=lifting,
        )
        faces = tuple(
            ExactFace(boundary.field, boundary.axis, side, boundary.value, boundary.value_function)
            for boundary in settings.boundaries
            for side in (("lower", "upper") if boundary.side == "both" else (boundary.side,))
        )
        source = (directory / "functions.py").read_bytes()
        if lifting is not None:
            definition = [
                (face.field, face.axis, face.side, face.value, face.value_function)
                for face in faces
            ]
            source += json.dumps(sorted(definition), sort_keys=True).encode()
        signature = hashlib.sha256(source).hexdigest()
        return cls(
            settings.training.constraints,
            tuple(problem.fields),
            faces,
            ("equations",),
            signature,
            recipe,
            hard_executor(recipe, settings.training.hard_execution),
        )

    @classmethod
    def _robin(
        cls, settings: TaskSettings, problem: Problem, adapter: ProblemAdapter, directory: Path
    ) -> ConstraintPlan:
        robin = compile_robin(settings, problem, adapter)
        faces = tuple(
            ExactFace(
                b.field,
                b.axis,
                side,
                b.value,
                b.value_function,
                b.kind,
                b.alpha if b.kind == "robin" else 0,
                b.beta if b.kind == "robin" else 1,
            )
            for b in settings.boundaries
            for side in (("lower", "upper") if b.side == "both" else (b.side,))
        )
        robin_definition = [
            (f.field, f.axis, f.side, f.kind, f.alpha, f.beta, f.value, f.value_function)
            for f in faces
        ]
        source = (directory / "functions.py").read_bytes() + json.dumps(
            sorted(robin_definition)
        ).encode()
        return cls(
            settings.training.constraints,
            tuple(problem.fields),
            faces,
            ("equations",),
            hashlib.sha256(source).hexdigest(),
            robin,
            None,
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
        if boundary.kind != "dirichlet":
            raise ValueError("Hard constraints require Dirichlet faces")
        sides = ("lower", "upper") if boundary.side == "both" else (boundary.side,)
        keys = {(boundary.field, boundary.axis, side) for side in sides}
        if provided & keys:
            raise ValueError("Hard constraints require one condition per face")
        provided.update(keys)
    if provided != required:
        raise ValueError("Hard constraints require all faces for every physical field")
    if all(face.value == 0 and face.value_function is None for face in settings.boundaries):
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
