"""Expand compatible Robin data by boundary intersections, without recursive traces."""

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import cast

import torch

from pinnstorch.training.dirichlet_lifting import TraceFunction
from pinnstorch.training.robin_projection import IntervalRobinProjection


@dataclass(frozen=True)
class RobinLiftingTerm:
    faces: tuple[tuple[int, int], ...]

    @property
    def sign(self) -> int:
        return 1 if len(self.faces) % 2 else -1


@dataclass(frozen=True)
class ExpandedRobinLifting:
    axes: tuple[IntervalRobinProjection, ...]
    targets: tuple[tuple[TraceFunction, TraceFunction], ...]
    paired_time: bool = True
    terms: tuple[RobinLiftingTerm, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not 1 <= len(self.axes) <= 3 or len(self.targets) != len(self.axes):
            raise ValueError("Robin lifting requires matching data on one to three axes")
        object.__setattr__(
            self,
            "terms",
            tuple(
                RobinLiftingTerm(tuple(zip(axes, sides)))
                for degree in range(1, len(self.axes) + 1)
                for axes in combinations(range(len(self.axes)), degree)
                for sides in product((0, 1), repeat=degree)
            ),
        )

    def _trace(
        self,
        value: torch.Tensor,
        projected: list[torch.Tensor],
        faces: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        for axis, side in faces:
            coordinate = projected[axis]
            derivative = (
                torch.autograd.grad(
                    value,
                    coordinate,
                    torch.ones_like(value),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if value.requires_grad
                else None
            )
            derivative = coordinate.mul(0) if derivative is None else derivative
            operator = (self.axes[axis].lower, self.axes[axis].upper)[side]
            normal = -1 if side == 0 else 1
            derivative_term = derivative.mul(normal * operator.beta)
            value = (
                derivative_term
                if operator.alpha == 0
                else value.mul(operator.alpha).add(derivative_term)
            )
        return value

    def _difference_trace(
        self,
        target: TraceFunction,
        projected: list[torch.Tensor],
        time: torch.Tensor,
        initial_time: torch.Tensor,
        faces: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        if not self.paired_time:
            delta = target(projected, time).sub(target(projected, initial_time))
            return self._trace(delta, projected, faces)
        # Pointwise callbacks and linear operators allow one batch for both times.
        columns = [torch.cat((x, x)) for x in projected]
        clock = torch.cat((time, initial_time))
        trace = self._trace(target(columns, clock), columns, faces)
        return cast(torch.Tensor, trace[: len(time)].sub(trace[len(time) :]))

    def difference(
        self, spatial: list[torch.Tensor], time: torch.Tensor, initial_time: torch.Tensor
    ) -> torch.Tensor:
        if len(self.axes) == 1:
            return self._interval_difference(spatial[0], time, initial_time)
        bases = [axis.basis(x) for axis, x in zip(self.axes, spatial)]
        fixed = {
            (axis, side): torch.full_like(x, self.axes[axis].bounds[side], requires_grad=True)
            for axis, x in enumerate(spatial)
            for side in (0, 1)
        }
        geometry: dict[tuple[tuple[int, int], ...], tuple[list[torch.Tensor], torch.Tensor]] = {
            (): (spatial, torch.ones_like(time))
        }
        result = torch.zeros_like(time)
        for term in self.terms:
            parent, parent_weight = geometry[term.faces[:-1]]
            axis, side = term.faces[-1]
            projected = list(parent)
            projected[axis] = fixed[axis, side]
            weight = parent_weight.mul(bases[axis][side])
            geometry[term.faces] = (projected, weight)
            primary_axis, primary_side = term.faces[0]
            target = self.targets[primary_axis][primary_side]
            trace = self._difference_trace(target, projected, time, initial_time, term.faces[1:])
            result = result.add(weight.mul(trace).mul(term.sign))
        return result

    def _interval_difference(
        self, coordinate: torch.Tensor, time: torch.Tensor, initial_time: torch.Tensor
    ) -> torch.Tensor:
        # A_0=0: no intersection trace or paired-batch derivative graph is needed.
        basis = self.axes[0].basis(coordinate)
        result = torch.zeros_like(time)
        for side, target in enumerate(self.targets[0]):
            face = [torch.full_like(coordinate, self.axes[0].bounds[side])]
            delta = target(face, time).sub(target(face, initial_time))
            result = result.add(basis[side].mul(delta))
        return result
