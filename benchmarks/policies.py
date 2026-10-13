"""Grid, optimization budget and acceptance are independent benchmark policies."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class GridPolicy(Protocol):
    def domain(self, dimensions: int, evaluating: bool) -> dict[str, Any]: ...

    def spatial_bounds(self) -> tuple[float, float]: ...


@dataclass(frozen=True)
class CartesianGrid:
    space_points: tuple[int, int] = (16, 41)
    time_points: tuple[int, int] = (11, 31)
    space_interval: tuple[float, float] = (0, 1)
    time_interval: tuple[float, float] = (0, 0.5)

    def spatial_bounds(self) -> tuple[float, float]:
        return self.space_interval

    def domain(self, dimensions: int, evaluating: bool) -> dict[str, Any]:
        index = int(evaluating)
        return {
            "space": {
                axis: {"interval": list(self.space_interval), "points": self.space_points[index]}
                for axis in ("x", "y", "z")[:dimensions]
            },
            "time": {"interval": list(self.time_interval), "points": self.time_points[index]},
        }


@dataclass(frozen=True)
class TrainingBudget:
    layers: tuple[int, ...] = (32, 32, 32)
    batch_sizes: tuple[int, int] = (128, 2048)
    learning_rate: float = 0.001
    collocation_points: int = 1024

    def network(self) -> dict[str, Any]:
        return {"hidden_layers": list(self.layers)}

    def training(self, steps: int, seed: int, evaluating: bool) -> dict[str, Any]:
        return {
            "steps": steps,
            "epochs": steps,
            "seed": seed,
            "batch_size": self.batch_sizes[int(evaluating)],
            "optimizer": {"name": "adam", "learning_rate": self.learning_rate},
            "collocation": {"sampling": "lhs", "points": self.collocation_points},
        }


class QualityPolicy(Protocol):
    def failures(self, metrics: Mapping[str, float]) -> list[str]: ...


@dataclass(frozen=True)
class FixedQuality:
    relative_l2: float = 0.05
    max_error: float = 0.15
    pde_rmse: float = 0.1
    initial_rmse: float = 0.05
    boundary_rmse: float = 0.05

    def failures(self, metrics: Mapping[str, float]) -> list[str]:
        limits = {
            "relative_l2": self.relative_l2,
            "max_error": self.max_error,
            "pde_rmse": self.pde_rmse,
            "initial_rmse": self.initial_rmse,
            "boundary_rmse": self.boundary_rmse,
        }
        return [
            f"{name}={metrics[name]:.6g} exceeds {limit}"
            for name, limit in limits.items()
            if not math.isfinite(metrics[name]) or metrics[name] > limit
        ]
