"""Experimental product-rule derivatives; no detached training geometry or IC."""

from __future__ import annotations

from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, TensorFields
from pinnstorch.training.constraints import HardBoxInitial


class HardBoxDerivatives:
    """One forward's derivatives for u = g(x) + tau(t) B(x) N(x,t).

    Instances own an autograd graph and must never be reused across optimizer
    steps. Higher derivatives use the physical field's regular autograd path.
    """

    def __init__(
        self, transform: HardBoxInitial, network: TensorFields, coordinates: Coordinates
    ) -> None:
        if coordinates.parameters or len(coordinates.spatial) != len(transform.bounds):
            raise ValueError("Product-rule hard requires aligned spatial axes without parameters")
        self.coordinates = coordinates
        self.network = Equation(network, coordinates)
        lower, upper = transform.time_interval
        self.time_scale = 1 / (upper - lower)
        self.tau = coordinates.time.sub(lower).div(upper - lower)
        initial = transform.background(list(coordinates.spatial.values()), coordinates.time)
        self.initial = Equation(initial, coordinates)
        self.time_dependent_background = transform.lifting is not None
        self.parts: list[torch.Tensor] = []
        normalized_columns: list[torch.Tensor] = []
        for column, (left, right) in zip(coordinates.spatial.values(), transform.bounds):
            normalized = column.sub(left).div(right - left)
            normalized_columns.append(normalized)
            self.parts.append(normalized.mul(normalized.neg().add(1)).mul(4))
        self.basis = self._product()
        self._geometry: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for index, ((axis, column), (left, right)) in enumerate(
            zip(coordinates.spatial.items(), transform.bounds)
        ):
            other = self._product(exclude=index)
            length = right - left
            first = other.mul(4 / length).mul(normalized_columns[index].mul(-2).add(1))
            second = other.mul(-8 / length**2)
            self._geometry[axis] = (first, second)

    def _product(self, exclude: int | None = None) -> torch.Tensor:
        result = torch.ones_like(self.coordinates.time)
        for index, part in enumerate(self.parts):
            if index != exclude:
                result = result.mul(part)
        return result

    def field(self, name: str) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.initial.field(name).add(self.basis.mul(self.tau).mul(self.network.field(name))),
        )

    def supported_diff(self, name: str, axis: str, order: int) -> torch.Tensor | None:
        if order not in (1, 2):
            return None
        return self.diff(name, axis, order)

    def diff(self, name: str, axis: str, order: int = 1) -> torch.Tensor:
        if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2):
            raise ValueError("Product-rule prototype supports derivative orders 1 and 2")
        value = self.network.field(name)
        first = self.network.diff(name, axis)
        if axis == "t" and self.time_dependent_background:
            correction = self.basis * (
                value * self.time_scale + self.tau * first
                if order == 1
                else 2 * self.time_scale * first + self.tau * self.network.diff(name, axis, 2)
            )
            return cast(torch.Tensor, self.initial.diff(name, axis, order) + correction)
        if axis == "t":
            if order == 1:
                return cast(
                    torch.Tensor,
                    self.basis.mul(value.mul(self.time_scale).add(self.tau.mul(first))),
                )
            return cast(
                torch.Tensor,
                self.basis.mul(
                    first.mul(2 * self.time_scale).add(
                        self.tau.mul(self.network.diff(name, axis, 2))
                    )
                ),
            )
        basis_first, basis_second = self._geometry[axis]
        if order == 1:
            correction = basis_first * value + self.basis * first
        else:
            correction = (
                basis_second * value
                + 2 * basis_first * first
                + self.basis * self.network.diff(name, axis, 2)
            )
        return cast(torch.Tensor, self.initial.diff(name, axis, order) + self.tau * correction)

    def heat_residual(self, name: str, conductivity: float) -> torch.Tensor:
        """Constant scalar heat only: u_t - kappa Delta(u)."""
        laplacian = torch.zeros_like(self.coordinates.time)
        for axis in self.coordinates.spatial:
            laplacian = laplacian + self.diff(name, axis, 2)
        return cast(torch.Tensor, self.diff(name, "t").sub(laplacian.mul(conductivity)))
