"""Deterministic sampling strategies isolated from global random generators."""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pinnstorch.contracts import FloatArray


@dataclass(frozen=True)
class TimePower:
    """Inverse CDF for density (power+1)*s**power on normalized physical time."""

    interval: tuple[float, float]
    power: float

    def __post_init__(self) -> None:
        lower, upper = self.interval
        if not all(math.isfinite(x) for x in (lower, upper, self.power)) or lower >= upper:
            raise ValueError("Time density requires finite increasing bounds and finite power")
        if self.power < 0:
            raise ValueError("Time density power must be nonnegative")

    def __call__(self, time: FloatArray) -> FloatArray:
        if self.power == 0:
            return time
        lower, upper = self.interval
        normalized = (time - lower) / (upper - lower)
        return np.asarray(
            lower + (upper - lower) * normalized ** (1 / (self.power + 1)), dtype=np.float64
        )


class SamplingPolicy(Protocol):
    @property
    def every_epochs(self) -> int: ...

    def points(self, epoch: int) -> FloatArray: ...

    def signature(self) -> tuple[int, int, int]: ...


@dataclass(frozen=True)
class ResampledLHS:
    count: int
    lower: FloatArray
    upper: FloatArray
    seed: int
    every_epochs: int

    def signature(self) -> tuple[int, int, int]:
        return (self.count, self.seed, self.every_epochs)

    def points(self, epoch: int) -> FloatArray:
        generation = epoch // self.every_epochs
        random = np.random.default_rng(np.random.SeedSequence([self.seed, generation]))
        unit = np.column_stack(
            [
                (random.permutation(self.count) + random.random(self.count)) / self.count
                for _ in self.lower
            ]
        )
        return np.asarray(self.lower + (self.upper - self.lower) * unit, dtype=np.float64)
