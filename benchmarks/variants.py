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
class RateDecay:
    final_ratio: float = 1.0

    def apply(self, config: dict[str, Any]) -> None:
        if self.final_ratio != 1:
            config["training"]["schedule"] = {
                "kind": "exponential",
                "final_ratio": self.final_ratio,
            }


@dataclass(frozen=True)
class CollocationRefresh:
    every_epochs: int = 0
    points: int | None = None
    time_power: float = 0.0

    def apply(self, config: dict[str, Any]) -> None:
        if self.points is not None:
            config["training"]["collocation"]["points"] = self.points
        if self.every_epochs:
            config["training"]["collocation"]["resample_every_epochs"] = self.every_epochs
        if self.time_power:
            config["training"]["collocation"]["time_power"] = self.time_power


@dataclass(frozen=True)
class RefinementSteps:
    steps: int = 0

    def apply(self, config: dict[str, Any]) -> None:
        if self.steps:
            config["training"]["refinement"] = {"steps": self.steps}


@dataclass(frozen=True)
class Constraints:
    kind: str = "soft"

    def apply(self, config: dict[str, Any]) -> None:
        if self.kind != "soft":
            config["training"]["constraints"] = self.kind


@dataclass(frozen=True)
class HardExecution:
    backend: str = "eager"

    def apply(self, config: dict[str, Any]) -> None:
        if (
            self.backend != "eager"
            or config["training"].get("constraints") == "hard_initial_robin"
        ):
            config["training"]["hard_execution"] = self.backend


@dataclass(frozen=True)
class HardLifting:
    backend: str = "recursive"

    def apply(self, config: dict[str, Any]) -> None:
        if (
            self.backend != "recursive"
            or config["training"].get("constraints") == "hard_initial_robin"
        ):
            config["training"]["hard_lifting"] = self.backend


@dataclass(frozen=True)
class TrainingComposition:
    adapters: tuple[TrainingAdapter, ...]

    def apply(self, config: dict[str, Any]) -> None:
        for adapter in self.adapters:
            adapter.apply(config)


@dataclass(frozen=True)
class BenchmarkVariant(BenchmarkCase):
    base: BenchmarkCase = field(kw_only=True)
    adapter: TrainingAdapter = field(kw_only=True)

    def __post_init__(self) -> None:
        # Proxy metadata belongs to the base PDE, not to adapter defaults.
        object.__setattr__(self, "grid", self.base.grid)
        object.__setattr__(self, "budget", self.base.budget)
        object.__setattr__(self, "quality", self.base.quality)

    def reference(self, spatial: dict[str, Array], time: Array) -> Array:
        return self.base.reference(spatial, time)

    def additional_errors(
        self, spatial: dict[str, Array], time: Array, error: Array
    ) -> dict[str, float]:
        return self.base.additional_errors(spatial, time, error)

    def boundary_mask(self, spatial: dict[str, Array]) -> NDArray[np.bool_]:
        return self.base.boundary_mask(spatial)

    def config(
        self, device: str, steps: int, seed: int, evaluating: bool = False
    ) -> dict[str, Any]:
        config = self.base.config(device, steps, seed, evaluating)
        self.adapter.apply(config)
        return config
