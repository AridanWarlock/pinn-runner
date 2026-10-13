"""Hermite projection onto two homogeneous normal-derivative boundary operators."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch

from pinnstorch.contracts import TensorOutputs

IntervalField = Callable[[torch.Tensor, torch.Tensor], TensorOutputs]


@dataclass(frozen=True)
class RobinOperator:
    """alpha*u + beta*du/dn; beta refers to a derivative, not physical flux."""

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or not math.isfinite(self.beta) or self.beta == 0:
            raise ValueError("Robin projection requires finite coefficients and nonzero beta")


@dataclass(frozen=True)
class IntervalRobinProjection:
    bounds: tuple[float, float]
    lower: RobinOperator
    upper: RobinOperator

    def __post_init__(self) -> None:
        left, right = self.bounds
        if not math.isfinite(left) or not math.isfinite(right) or right <= left:
            raise ValueError("Projection requires a finite positive interval length")

    def basis(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = self.bounds
        length = right - left
        s = x.sub(left).div(length)
        # Cubic Hermite derivative bases; both vanish at both endpoints.
        lower = s.mul(s.neg().add(1).square()).mul(-length / self.lower.beta)
        upper = s.square().mul(s.sub(1)).mul(length / self.upper.beta)
        return lower, upper

    def traces(
        self, field: IntervalField, x: torch.Tensor, time: torch.Tensor
    ) -> tuple[TensorOutputs, TensorOutputs]:
        values: list[TensorOutputs] = []
        for side, operator in enumerate((self.lower, self.upper)):
            coordinate = torch.full_like(x, self.bounds[side], requires_grad=True)
            outputs = field(coordinate, time)
            traces: TensorOutputs = {}
            for name, value in outputs.items():
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
                normal = -1 if side == 0 else 1
                traces[name] = value.mul(operator.alpha).add(
                    derivative.mul(normal * operator.beta)
                )
            values.append(traces)
        return values[0], values[1]

    def __call__(self, field: IntervalField, x: torch.Tensor, time: torch.Tensor) -> TensorOutputs:
        # Traces need normal derivatives even when the caller only predicts.
        # Keep coordinate/parameter graphs when training; detach only no-grad output.
        tracking = torch.is_grad_enabled()
        with torch.set_grad_enabled(True):
            lower, upper = self.traces(field, x, time)
            basis_lower, basis_upper = self.basis(x)
            outputs = field(x, time)
            result = {
                name: cast(
                    torch.Tensor, value - basis_lower * lower[name] - basis_upper * upper[name]
                )
                for name, value in outputs.items()
            }
        return result if tracking else {name: value.detach() for name, value in result.items()}
