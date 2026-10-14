"""Batch equal boundary strata and apply only their necessary normal operators."""

from dataclasses import dataclass, field
from itertools import combinations, product

import torch

from pinnstorch.contracts import TensorOutputs
from pinnstorch.training.robin_box import BoxRobinProjection
from pinnstorch.training.robin_initial import NetworkField


@dataclass(frozen=True)
class BoundaryGroup:
    axes: tuple[int, ...]
    sides: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class GroupedBoxRobinProjection(BoxRobinProjection):
    groups: tuple[BoundaryGroup, ...] = field(init=False)
    _constants: dict[
        tuple[tuple[int, ...], int, torch.device, torch.dtype],
        tuple[torch.Tensor | None, torch.Tensor],
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        groups = tuple(
            BoundaryGroup(axes, tuple(product((0, 1), repeat=degree)))
            for degree in range(1, len(self.axes) + 1)
            for axes in combinations(range(len(self.axes)), degree)
        )
        object.__setattr__(self, "groups", groups)

    def _batch(
        self,
        group: BoundaryGroup,
        spatial: list[torch.Tensor],
        time: torch.Tensor,
        fixed: dict[tuple[int, int], torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        indices = {axis: index for index, axis in enumerate(group.axes)}
        columns = [
            torch.cat(
                [
                    fixed[axis, sides[indices[axis]]] if axis in indices else x
                    for sides in group.sides
                ]
            )
            for axis, x in enumerate(spatial)
        ]
        return columns, torch.cat([time] * len(group.sides))

    def _coefficients(
        self,
        group: BoundaryGroup,
        index: int,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        key = (group.axes, index, value.device, value.dtype)
        if key not in self._constants:
            axis = group.axes[index]
            operators = [
                (self.axes[axis].lower, self.axes[axis].upper)[sides[index]]
                for sides in group.sides
            ]
            beta = value.new_tensor(
                [
                    operator.beta * (-1 if sides[index] == 0 else 1)
                    for operator, sides in zip(operators, group.sides)
                ]
            ).reshape(-1, 1, 1)
            alpha = (
                None
                if all(operator.alpha == 0 for operator in operators)
                else value.new_tensor([operator.alpha for operator in operators]).reshape(-1, 1, 1)
            )
            self._constants[key] = (alpha, beta)
        return self._constants[key]

    def _trace(
        self,
        group: BoundaryGroup,
        value: torch.Tensor,
        columns: list[torch.Tensor],
        count: int,
        differentiable: bool,
    ) -> torch.Tensor:
        for index, axis in enumerate(group.axes):
            derivative = (
                torch.autograd.grad(
                    value,
                    columns[axis],
                    torch.ones_like(value),
                    create_graph=differentiable or index < len(group.axes) - 1,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if value.requires_grad
                else None
            )
            derivative = columns[axis].mul(0) if derivative is None else derivative
            alpha, beta = self._coefficients(group, index, value)
            derivative_term = derivative.reshape(len(group.sides), count, 1) * beta
            if alpha is None:
                # Fixed mathematical zero, not a threshold inferred from samples.
                value = derivative_term.reshape(-1, 1)
            else:
                value = (
                    value.reshape(len(group.sides), count, 1) * alpha + derivative_term
                ).reshape(-1, 1)
        return value if differentiable else value.detach()

    def apply(
        self,
        outputs: TensorOutputs,
        field: NetworkField,
        spatial: list[torch.Tensor],
        time: torch.Tensor,
    ) -> TensorOutputs:
        tracking = torch.is_grad_enabled()
        with torch.set_grad_enabled(True):
            fixed = {
                (axis, side): torch.full_like(x, self.axes[axis].bounds[side], requires_grad=True)
                for axis, x in enumerate(spatial)
                for side in (0, 1)
            }
            bases = [axis.basis(x) for axis, x in zip(self.axes, spatial)]
            result = dict(outputs)
            for group in self.groups:
                columns, clock = self._batch(group, spatial, time, fixed)
                boundary = field(columns, clock)
                weights = torch.ones(
                    (len(group.sides), *time.shape), dtype=time.dtype, device=time.device
                )
                for index, axis in enumerate(group.axes):
                    weights = weights * torch.stack(
                        [bases[axis][sides[index]] for sides in group.sides]
                    )
                for name, value in boundary.items():
                    trace = self._trace(group, value, columns, len(time), tracking)
                    correction = (weights * trace.reshape(len(group.sides), *time.shape)).sum(
                        dim=0
                    )
                    result[name] = result[name].add(correction.mul((-1) ** len(group.axes)))
        return result if tracking else {name: value.detach() for name, value in result.items()}
