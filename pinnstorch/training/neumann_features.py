"""Initial/background field plus a network with homogeneous Neumann coordinate features."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from pinnstorch.contracts import TensorOutputs

from .robin_background import InitialOnlyBackground
from .robin_box import BoxRobinInitial
from .robin_initial import IntervalRobinInitial


@dataclass(frozen=True)
class FeatureNeumannInitial:
    requires_coordinate_grad: ClassVar[bool] = False
    recipe: IntervalRobinInitial | BoxRobinInitial

    @property
    def static_background(self) -> InitialOnlyBackground | None:
        return self.recipe.static_background

    def __call__(self, outputs: TensorOutputs, *columns: torch.Tensor) -> TensorOutputs:
        spatial, time = list(columns[:-1]), columns[-1]
        tracking = torch.is_grad_enabled()
        # General continuation can differentiate boundary data at face intersections.
        with torch.set_grad_enabled(tracking or self.static_background is None):
            background = (
                self.recipe.background(spatial[0], time)
                if isinstance(self.recipe, IntervalRobinInitial)
                else self.recipe.background(spatial, time)
            )
            lower, upper = self.recipe.time_interval
            tau = time.sub(lower).div(upper - lower)
            result = {name: value + tau * outputs[name] for name, value in background.items()}
        return result if tracking else {name: value.detach() for name, value in result.items()}
