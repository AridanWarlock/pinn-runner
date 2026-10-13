"""Deterministic sampling strategies isolated from global random generators."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pinnstorch.contracts import FloatArray


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
