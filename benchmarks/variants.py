"""Compose a benchmark with a training experiment, preserving its PDE identity."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .cases import Array, BenchmarkCase


class TrainingAdapter(Protocol):
    def apply(self, config: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class BoundaryWeight:
    weight: float
    gradient_norm_every: int = 0

    def apply(self, config: dict[str, Any]) -> None:
        if self.weight != 1 or self.gradient_norm_every:
            config["training"]["loss"] = {
                "kind": "fixed" if self.weight != 1 else "equal",
                "boundaries": self.weight,
                "gradient_norm_every": self.gradient_norm_every,
            }


@dataclass(frozen=True)
class BenchmarkVariant(BenchmarkCase):
    base: BenchmarkCase = field(kw_only=True)
    adapter: TrainingAdapter = field(kw_only=True)

    def reference(self, spatial: dict[str, Array], time: Array) -> Array:
        return self.base.reference(spatial, time)

    def boundary_mask(self, spatial: dict[str, Array]) -> NDArray[np.bool_]:
        return self.base.boundary_mask(spatial)

    def config(
        self, device: str, steps: int, seed: int, evaluating: bool = False
    ) -> dict[str, Any]:
        config = self.base.config(device, steps, seed, evaluating)
        self.adapter.apply(config)
        return config
