"""Independent physical operators on sampled face interiors, without an analytical teacher."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from pinnstorch.api import Equation
from pinnstorch.contracts import TensorOutputs
from pinnstorch.runner.problem import ProblemAdapter
from pinnstorch.runner.settings import Boundary, TaskSettings


@dataclass(frozen=True)
class BoundaryOperatorError:
    field: str
    kind: str
    axis: str
    side: str
    points: int
    rmse: float
    max_error: float


def operator_value(equation: Equation, boundary: Boundary, normal: int) -> torch.Tensor:
    value = equation.field(boundary.field)
    if boundary.kind == "dirichlet":
        return value
    derivative: torch.Tensor = equation.diff(boundary.field, boundary.axis).mul(normal)
    if boundary.kind == "neumann":
        return derivative
    if boundary.kind == "robin":
        return cast(torch.Tensor, value.mul(boundary.alpha).add(derivative.mul(boundary.beta)))
    if boundary.kind != "heat_flux":
        raise ValueError(f"Unsupported boundary operator: {boundary.kind}")
    conductivity = boundary.conductivity
    if isinstance(conductivity, str):
        coordinates = equation.coordinates
        coefficient = (
            coordinates.parameters[conductivity]
            if conductivity in coordinates.parameters
            else coordinates.coefficients[conductivity]
        )
        if (
            coefficient.shape != value.shape
            or not torch.isfinite(coefficient).all()
            or not torch.all(coefficient.gt(0))
        ):
            raise ValueError("Heat-flux conductivity must be a finite positive column")
    else:
        coefficient = value.new_tensor(conductivity)
    return cast(torch.Tensor, derivative.neg().mul(coefficient))


@dataclass(frozen=True)
class FaceQuery:
    boundary: Boundary
    side: str
    spatial: list[torch.Tensor]
    clock: torch.Tensor

    def evaluate(
        self,
        adapter: ProblemAdapter,
        field: Callable[[list[torch.Tensor], torch.Tensor], TensorOutputs],
    ) -> BoundaryOperatorError:
        boundary, spatial, clock = self.boundary, self.spatial, self.clock
        with torch.set_grad_enabled(True):
            physical = dict(field(spatial, clock))
            equation = Equation(physical, adapter.context(spatial, clock))
            value = equation.field(boundary.field)
            if value.shape != clock.shape or not torch.isfinite(value).all():
                raise ValueError("Boundary physical field must be a finite scalar column")
            observed = operator_value(equation, boundary, -1 if self.side == "lower" else 1)
            target = (
                adapter.boundary_value(boundary.value_function, spatial, clock)
                if boundary.value_function is not None
                else torch.full_like(clock, cast(float, boundary.value))
            )
            if target.ndim == 0:
                target = target.expand_as(clock)
            if target.shape != clock.shape or observed.shape != clock.shape:
                raise ValueError("Boundary values and operators must be scalar columns")
            error = observed.detach() - target.detach()
        if not torch.isfinite(error).all():
            raise ValueError("Nonfinite boundary operator error")
        return BoundaryOperatorError(
            boundary.field,
            boundary.kind,
            boundary.axis,
            self.side,
            len(clock),
            float(error.square().mean().sqrt()),
            float(error.abs().max()),
        )


def sample_face(
    settings: TaskSettings,
    boundary: Boundary,
    side: str,
    generator: np.random.Generator,
    device: str,
    count: int,
    dtype: torch.dtype,
) -> FaceQuery:
    axes = (
        *settings.domain.space.values(),
        *settings.domain.parameters.values(),
        settings.domain.time,
    )
    names = tuple(settings.domain.space)
    axis = names.index(boundary.axis)
    samples = generator.uniform(0, 1, (count, len(axes)))
    columns = [
        torch.tensor(lower + samples[:, i : i + 1] * (upper - lower), dtype=dtype, device=device)
        for i, definition in enumerate(axes)
        for lower, upper in (definition.interval,)
    ]
    for index, definition in enumerate(axes[: len(names)]):
        lower, upper = (columns[index].new_tensor(x) for x in definition.interval)
        if index == axis:
            columns[index].fill_(lower if side == "lower" else upper)
        else:
            interior_lower, interior_upper = (
                torch.nextafter(lower, upper),
                torch.nextafter(upper, lower),
            )
            if interior_lower > interior_upper:
                raise ValueError("Face interiors are not representable in this dtype")
            columns[index].clamp_(interior_lower, interior_upper)
    columns[-1][0], columns[-1][1] = settings.domain.time.interval
    columns = [column.detach().requires_grad_() for column in columns]
    return FaceQuery(boundary, side, columns[:-1], columns[-1])


def boundary_operator_errors(
    settings: TaskSettings,
    adapter: ProblemAdapter,
    field: Callable[[list[torch.Tensor], torch.Tensor], TensorOutputs],
    device: str,
    points_per_face: int = 64,
    seed: int = 20261016,
    dtype: torch.dtype = torch.float32,
) -> tuple[BoundaryOperatorError, ...]:
    if points_per_face < 2:
        raise ValueError("At least two points per face required")
    generator = np.random.default_rng(seed)
    return tuple(
        sample_face(settings, boundary, side, generator, device, points_per_face, dtype).evaluate(
            adapter, field
        )
        for boundary in settings.boundaries
        for side in (("lower", "upper") if boundary.side == "both" else (boundary.side,))
    )
