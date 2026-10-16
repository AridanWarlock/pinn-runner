"""An initial-only background for explicitly time-independent operator data."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from pinnstorch.contracts import TensorOutputs


@dataclass(frozen=True)
class InitialOnlyBackground:
    initial: Callable[[list[torch.Tensor], torch.Tensor], TensorOutputs]
    time_origin: float

    def __call__(self, spatial: list[torch.Tensor], time: torch.Tensor) -> TensorOutputs:
        return self.initial(spatial, torch.full_like(time, self.time_origin))
