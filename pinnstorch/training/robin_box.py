"""Commuting Cartesian Robin projections and compatible operator-data lifting."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from pinnstorch.contracts import TensorOutputs
from pinnstorch.training.dirichlet_lifting import TraceFunction
from pinnstorch.training.robin_initial import NetworkField
from pinnstorch.training.robin_projection import IntervalRobinProjection


@dataclass(frozen=True)
class BoxRobinProjection:
    axes: tuple[IntervalRobinProjection, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.axes) <= 3:
            raise ValueError("Robin box requires one to three spatial axes")

    def traces(
        self, field: NetworkField, spatial: list[torch.Tensor], time: torch.Tensor, axis: int
    ) -> tuple[TensorOutputs, TensorOutputs]:
        def interval(coordinate: torch.Tensor, clock: torch.Tensor) -> TensorOutputs:
            projected = list(spatial)
            projected[axis] = coordinate
            return field(projected, clock)

        return self.axes[axis].traces(interval, spatial[axis], time)

    def apply(
        self,
        outputs: TensorOutputs,
        field: NetworkField,
        spatial: list[torch.Tensor],
        time: torch.Tensor,
    ) -> TensorOutputs:
        tracking = torch.is_grad_enabled()
        with torch.set_grad_enabled(True):
            result = self._apply(len(self.axes), outputs, field, spatial, time)
        return result if tracking else {name: value.detach() for name, value in result.items()}

    def _apply(
        self,
        count: int,
        outputs: TensorOutputs,
        field: NetworkField,
        spatial: list[torch.Tensor],
        time: torch.Tensor,
    ) -> TensorOutputs:
        if count == 0:
            return outputs
        axis = count - 1
        center = self._apply(axis, outputs, field, spatial, time)

        def previous(projected: list[torch.Tensor], clock: torch.Tensor) -> TensorOutputs:
            return self._apply(axis, field(projected, clock), field, projected, clock)

        lower, upper = self.traces(previous, spatial, time, axis)
        phi_lower, phi_upper = self.axes[axis].basis(spatial[axis])
        return {
            name: value - phi_lower * lower[name] - phi_upper * upper[name]
            for name, value in center.items()
        }


@dataclass(frozen=True)
class BoxRobinInitial:
    projection: BoxRobinProjection
    time_interval: tuple[float, float]
    initial: NetworkField
    targets: tuple[tuple[TraceFunction, TraceFunction], ...]
    field: str

    def bind(self, network: NetworkField) -> "BoundBoxRobinInitial":
        return BoundBoxRobinInitial(self, network)

    def _lifting(
        self,
        count: int,
        spatial: list[torch.Tensor],
        time: torch.Tensor,
        initial_time: torch.Tensor,
    ) -> TensorOutputs:
        if count == 0:
            return {self.field: torch.zeros_like(time)}
        axis = count - 1
        center = self._lifting(axis, spatial, time, initial_time)

        def previous(projected: list[torch.Tensor], clock: torch.Tensor) -> TensorOutputs:
            return self._lifting(axis, projected, clock, initial_time)

        traces = self.projection.traces(previous, spatial, time, axis)
        bases = self.projection.axes[axis].basis(spatial[axis])
        value = center[self.field]
        for side, target in enumerate(self.targets[axis]):
            face = list(spatial)
            face[axis] = torch.full_like(spatial[axis], self.projection.axes[axis].bounds[side])
            delta = target(face, time).sub(target(face, initial_time))
            value = value.add(bases[side].mul(delta.sub(traces[side][self.field])))
        return {self.field: value}

    def background(self, spatial: list[torch.Tensor], time: torch.Tensor) -> TensorOutputs:
        initial_time = torch.full_like(time, self.time_interval[0])
        initial = self.initial(spatial, initial_time)
        lifting = self._lifting(len(self.projection.axes), spatial, time, initial_time)
        return {self.field: initial[self.field].add(lifting[self.field])}


@dataclass(frozen=True)
class BoundBoxRobinInitial:
    requires_coordinate_grad: ClassVar[bool] = False
    recipe: BoxRobinInitial
    network: NetworkField

    def __call__(self, outputs: TensorOutputs, *columns: torch.Tensor) -> TensorOutputs:
        spatial, time = list(columns[:-1]), columns[-1]
        tracking = torch.is_grad_enabled()
        with torch.set_grad_enabled(True):
            correction = self.recipe.projection.apply(outputs, self.network, spatial, time)
            background = self.recipe.background(spatial, time)
            lower, upper = self.recipe.time_interval
            value = background[self.recipe.field].add(
                time.sub(lower).div(upper - lower).mul(correction[self.recipe.field])
            )
        return {self.recipe.field: value if tracking else value.detach()}
