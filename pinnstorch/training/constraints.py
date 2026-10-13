"""Exact initial values and zero Dirichlet faces without a solution teacher."""

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
    fused_predict: bool = False
    product_rule: bool = False
    spatial_names: tuple[str, ...] = ()

    def __call__(self, outputs: TensorOutputs, *columns: torch.Tensor) -> TensorOutputs:
        if self.product_rule and torch.is_grad_enabled():
            from pinnstorch.api import Coordinates, DerivativeFields
            from pinnstorch.training.hard_derivatives import HardBoxDerivatives

            names = self.spatial_names or ("x", "y", "z")[: len(self.bounds)]
            coordinates = Coordinates(dict(zip(names, columns[:-1])), columns[-1])
            derivatives = HardBoxDerivatives(self, outputs, coordinates)
            return DerivativeFields(
                {name: derivatives.field(name) for name in outputs}, derivatives.supported_diff
            )
        spatial, time = list(columns[:-1]), columns[-1]
        lower, upper = self.time_interval
        initial = self.initial(spatial, torch.full_like(time, lower))
        factor = torch.ones_like(time)
        for column, (left, right) in zip(spatial, self.bounds):
            normalized = column.sub(left).div(right - left)
            # Training keeps the original AD graph. addcmul reduces forward
            # launches, but its backward was slower on the current CUDA stack.
            term = (
                torch.addcmul(normalized, normalized, normalized, value=-1)
                if self.fused_predict and not torch.is_grad_enabled()
                else normalized.mul(normalized.neg().add(1))
            )
            factor = factor.mul(term.mul(4))
        factor = factor.mul(time.sub(lower).div(upper - lower))
        return {name: initial[name] + factor.mul(value) for name, value in outputs.items()}
