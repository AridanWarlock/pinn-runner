"""Execution adapters for one mathematical hard recipe; resolve before fitting."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from pinnstorch.api import Coordinates, DerivativeFields
from pinnstorch.contracts import TensorOutputs
from pinnstorch.training.constraints import HardBoxInitial
from pinnstorch.training.hard_derivatives import HardBoxDerivatives


@dataclass(frozen=True)
class ProductRuleHard:
    recipe: HardBoxInitial
    requires_coordinate_grad: ClassVar[bool] = False

    def __call__(self, outputs: TensorOutputs, *columns: torch.Tensor) -> TensorOutputs:
        if not torch.is_grad_enabled():
            return self.recipe(outputs, *columns)
        names = self.recipe.spatial_names or ("x", "y", "z")[: len(self.recipe.bounds)]
        coordinates = Coordinates(dict(zip(names, columns[:-1])), columns[-1])
        derivatives = HardBoxDerivatives(self.recipe, outputs, coordinates)
        return DerivativeFields(
            {name: derivatives.field(name) for name in outputs}, derivatives.supported_diff
        )


@dataclass(frozen=True)
class FusedPredictHard:
    recipe: HardBoxInitial
    requires_coordinate_grad: ClassVar[bool] = False

    def __call__(self, outputs: TensorOutputs, *columns: torch.Tensor) -> TensorOutputs:
        if torch.is_grad_enabled():
            return self.recipe(outputs, *columns)
        spatial, time = list(columns[:-1]), columns[-1]
        lower, upper = self.recipe.time_interval
        initial = self.recipe.initial(spatial, torch.full_like(time, lower))
        factor = torch.ones_like(time)
        for column, (left, right) in zip(spatial, self.recipe.bounds):
            normalized = column.sub(left).div(right - left)
            term = torch.addcmul(normalized, normalized, normalized, value=-1)
            factor = factor.mul(term.mul(4))
        factor = factor.mul(time.sub(lower).div(upper - lower))
        return {name: initial[name] + factor.mul(value) for name, value in outputs.items()}


HardExecutor = HardBoxInitial | FusedPredictHard | ProductRuleHard


def hard_executor(recipe: HardBoxInitial, backend: str) -> HardExecutor:
    if backend == "eager":
        return recipe
    if backend == "fused_predict":
        return FusedPredictHard(recipe)
    if backend == "product_rule":
        return ProductRuleHard(recipe)
    raise ValueError(f"Unsupported hard execution '{backend}'")


def hard_recipe(transform: object) -> HardBoxInitial:
    if isinstance(transform, HardBoxInitial):
        return transform
    if isinstance(transform, (FusedPredictHard, ProductRuleHard)):
        return transform.recipe
    raise ValueError("Checkpoint must use a box-initial hard recipe")
