"""Exact initial values and compatible Dirichlet faces without a solution teacher."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch

from pinnstorch.contracts import TensorOutputs


@dataclass(frozen=True)
class HardBoxInitial:
    requires_coordinate_grad: ClassVar[bool] = False
    bounds: Sequence[tuple[float, float]]
    time_interval: tuple[float, float]
    initial: Callable[[list[torch.Tensor], torch.Tensor], TensorOutputs]
    spatial_names: tuple[str, ...] = ()
    lifting: Callable[[list[torch.Tensor], torch.Tensor], TensorOutputs] | None = None

    def background(self, spatial: list[torch.Tensor], time: torch.Tensor) -> TensorOutputs:
        initial_time = torch.full_like(time, self.time_interval[0])
        values = self.initial(spatial, initial_time)
        if self.lifting is None:
            return values
        current = self.lifting(spatial, time)
        original = self.lifting(spatial, initial_time)
        return {name: value + (current[name] - original[name]) for name, value in values.items()}

    def __call__(self, outputs: TensorOutputs, *columns: torch.Tensor) -> TensorOutputs:
        spatial, time = list(columns[:-1]), columns[-1]
        lower, upper = self.time_interval
        initial = self.background(spatial, time)
        factor = torch.ones_like(time)
        for column, (left, right) in zip(spatial, self.bounds):
            normalized = column.sub(left).div(right - left)
            term = normalized.mul(normalized.neg().add(1))
            factor = factor.mul(term.mul(4))
        factor = factor.mul(time.sub(lower).div(upper - lower))
        return {name: initial[name] + factor.mul(value) for name, value in outputs.items()}
