"""Step-based learning-rate strategies, independent from Lightning."""

from dataclasses import dataclass
from typing import Protocol

import torch


class RatePolicy(Protocol):
    def build(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler | None: ...


@dataclass(frozen=True)
class ConstantRate:
    def build(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        return None


@dataclass(frozen=True)
class ExponentialRate:
    steps: int
    final_ratio: float

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.ExponentialLR:
        if self.steps < 1 or not 0 < self.final_ratio <= 1:
            raise ValueError("Exponential schedule needs positive steps and final_ratio in (0, 1]")
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, self.final_ratio ** (1 / self.steps)
        )
