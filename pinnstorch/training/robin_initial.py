"""Exact initial field and two time-dependent interval Robin/Neumann conditions."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import torch

from pinnstorch.contracts import TensorOutputs
from pinnstorch.training.dirichlet_lifting import TraceFunction
from pinnstorch.training.robin_background import InitialOnlyBackground
from pinnstorch.training.robin_projection import IntervalRobinProjection

NetworkField = Callable[[list[torch.Tensor], torch.Tensor], TensorOutputs]


@dataclass(frozen=True)
class IntervalRobinInitial:
    projection: IntervalRobinProjection
    time_interval: tuple[float, float]
    initial: NetworkField
    targets: tuple[TraceFunction, TraceFunction]
    field: str
    static_background: InitialOnlyBackground | None = None

    def bind(self, network: NetworkField) -> "BoundRobinInitial":
        return BoundRobinInitial(self, network)

    def background(self, x: torch.Tensor, time: torch.Tensor) -> TensorOutputs:
        if self.static_background is not None:
            return self.static_background([x], time)
        initial_time = torch.full_like(time, self.time_interval[0])
        result = self.initial([x], initial_time)
        basis = self.projection.basis(x)
        value = result[self.field]
        for side, target in enumerate(self.targets):
            face = torch.full_like(x, self.projection.bounds[side])
            delta = target([face], time).sub(target([face], initial_time))
            value = value.add(basis[side].mul(delta))
        return {self.field: value}


@dataclass(frozen=True)
class BoundRobinInitial:
    requires_coordinate_grad: ClassVar[bool] = False
    recipe: IntervalRobinInitial
    network: NetworkField

    def __call__(self, outputs: TensorOutputs, *columns: torch.Tensor) -> TensorOutputs:
        x, time = columns
        tracking = torch.is_grad_enabled()
        with torch.set_grad_enabled(True):

            def field(coordinate: torch.Tensor, clock: torch.Tensor) -> TensorOutputs:
                return self.network([coordinate], clock)

            correction = self.recipe.projection.apply(outputs, field, x, time)
            background = self.recipe.background(x, time)
            lower, upper = self.recipe.time_interval
            tau = time.sub(lower).div(upper - lower)
            result = {
                name: value.add(tau.mul(correction[name])) for name, value in background.items()
            }
        return result if tracking else {name: value.detach() for name, value in result.items()}
