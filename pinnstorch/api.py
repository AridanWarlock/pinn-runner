"""Public language for defining equations, independent of training machinery."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import torch

TensorFields = dict[str, torch.Tensor]


@dataclass(frozen=True)
class Coordinates:
    """Named physical coordinates and parameters for one vectorized batch."""

    spatial: Mapping[str, torch.Tensor]
    time: torch.Tensor
    parameters: Mapping[str, torch.Tensor] = field(default_factory=dict)

    @property
    def x(self) -> torch.Tensor:
        return self.spatial["x"]

    @property
    def y(self) -> torch.Tensor:
        return self.spatial["y"]

    @property
    def z(self) -> torch.Tensor:
        return self.spatial["z"]

    @property
    def t(self) -> torch.Tensor:
        return self.time

    def variable(self, name: str) -> torch.Tensor:
        if name == "t":
            return self.time
        if name in self.spatial:
            return self.spatial[name]
        if name in self.parameters:
            return self.parameters[name]
        raise ValueError(f"Unknown coordinate or parameter '{name}'")


class Equation:
    """Fields and differentiable operators with a per-batch derivative cache."""

    def __init__(self, fields: Mapping[str, torch.Tensor], coordinates: Coordinates) -> None:
        self.fields = fields
        self.coordinates = coordinates
        self.parameters = coordinates.parameters
        self._derivatives: dict[tuple[str, tuple[str, ...]], torch.Tensor] = {}

    def field(self, name: str) -> torch.Tensor:
        try:
            return self.fields[name]
        except KeyError as error:
            raise ValueError(f"Model does not provide field '{name}'") from error

    def diff(self, value: str | torch.Tensor, axis: str, order: int = 1) -> torch.Tensor:
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("Derivative order must be a positive integer")
        coordinate = self.coordinates.variable(axis)
        if isinstance(value, str) and (value, (axis,)) not in self._derivatives:
            self._prime_first_derivatives(value)
        result = self.field(value) if isinstance(value, str) else value
        for degree in range(1, order + 1):
            key = (value, (axis,) * degree) if isinstance(value, str) else None
            if key is not None and key in self._derivatives:
                result = self._derivatives[key]
                continue
            result = self._differentiate(result, coordinate)
            if key is not None:
                self._derivatives[key] = result
        return result

    def _prime_first_derivatives(self, name: str) -> None:
        variables = {**self.coordinates.spatial, "t": self.coordinates.time}
        selected = {
            axis: variable for axis, variable in variables.items() if variable.requires_grad
        }
        value = self.field(name)
        if not selected:
            return
        derivatives = (
            torch.autograd.grad(
                value,
                list(selected.values()),
                torch.ones_like(value),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            if value.requires_grad
            else (None,) * len(selected)
        )
        for (axis, coordinate), derivative in zip(selected.items(), derivatives):
            self._derivatives[(name, (axis,))] = cast(
                torch.Tensor, coordinate * 0 if derivative is None else derivative
            )

    @staticmethod
    def _differentiate(value: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        if not coordinate.requires_grad:
            raise ValueError("Derivatives require coordinates with requires_grad=True")
        derivative = None
        if value.requires_grad:
            derivative = torch.autograd.grad(
                value,
                coordinate,
                torch.ones_like(value),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
        return cast(torch.Tensor, coordinate * 0 if derivative is None else derivative)

    def dt(self, field: str) -> torch.Tensor:
        return self.diff(field, "t")

    def gradient(self, field: str) -> TensorFields:
        return {axis: self.diff(field, axis) for axis in self.coordinates.spatial}

    def laplacian(self, field: str) -> torch.Tensor:
        derivatives = [self.diff(field, axis, 2) for axis in self.coordinates.spatial]
        return self._sum(derivatives)

    def divergence(self, vector: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if set(vector) != set(self.coordinates.spatial):
            raise ValueError("Divergence requires one component for each physical spatial axis")
        derivatives = [self.diff(vector[axis], axis) for axis in self.coordinates.spatial]
        return self._sum(derivatives)

    @staticmethod
    def _sum(terms: Sequence[torch.Tensor]) -> torch.Tensor:
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result


EquationFunction = Callable[[Equation], torch.Tensor]
InitialFunction = Callable[[Coordinates], TensorFields]
ValueFunction = Callable[[Coordinates], torch.Tensor | float]
TransformFunction = Callable[[Equation], TensorFields]


@dataclass(frozen=True)
class Problem:
    """A family of physical fields, equations, initial and boundary values."""

    fields: Sequence[str]
    equations: Mapping[str, EquationFunction]
    initial: InitialFunction
    boundary_values: Mapping[str, ValueFunction] = field(default_factory=dict)
    transform: TransformFunction | None = None
    network_fields: Sequence[str] | None = None

    def __post_init__(self) -> None:
        groups = (
            self.fields,
            self.equations,
            self.network_fields if self.network_fields is not None else self.fields,
        )
        for names in groups:
            if not names or len(set(names)) != len(names):
                raise ValueError("Problem field/equation names must be nonempty and unique")
            if any(
                not isinstance(name, str)
                or not name.isidentifier()
                or name.startswith("_")
                or not name.isascii()
                for name in names
            ):
                raise ValueError(
                    "Problem names must be ASCII identifiers without leading underscores"
                )
        if self.transform is None and not set(self.fields) <= set(self.outputs):
            raise ValueError("Different physical/network fields require Problem.transform")
        if any(not callable(function) for function in self.boundary_values.values()):
            raise ValueError("Problem.boundary_values must contain callable functions")
        if set(self.equations) & (set(self.fields) | set(self.outputs)):
            raise ValueError("Equation names must differ from field names")
        if not callable(self.initial) or any(
            not callable(function) for function in self.equations.values()
        ):
            raise ValueError("Problem.initial and equations must be callable")
        if self.transform is not None and not callable(self.transform):
            raise ValueError("Problem.transform must be callable")

    @property
    def outputs(self) -> tuple[str, ...]:
        return tuple(self.network_fields if self.network_fields is not None else self.fields)
