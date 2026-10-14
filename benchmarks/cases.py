"""Fixed problem definitions and independent NumPy analytical references."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .policies import CartesianGrid, FixedQuality, GridPolicy, QualityPolicy, TrainingBudget

Array = NDArray[np.float64]
PROBLEMS = Path(__file__).parent / "problems"


@dataclass(frozen=True)
class BenchmarkCase(ABC):
    name: str
    dimensions: int
    grid: GridPolicy = CartesianGrid()
    budget: TrainingBudget = TrainingBudget()
    quality: QualityPolicy = FixedQuality()

    @property
    def source(self) -> Path:
        return PROBLEMS / self.name / "functions.py"

    @abstractmethod
    def reference(self, spatial: dict[str, Array], time: Array) -> Array:
        """Independent analytical solution, with NumPy broadcasting."""

    def additional_errors(
        self, spatial: dict[str, Array], time: Array, error: Array
    ) -> dict[str, float]:
        return {}

    def boundaries(self, axes: tuple[str, ...]) -> list[dict[str, Any]]:
        return [{"kind": "dirichlet", "field": "u", "axis": axis, "value": 0} for axis in axes]

    def boundary_mask(self, spatial: dict[str, Array]) -> NDArray[np.bool_]:
        mask = np.zeros(len(next(iter(spatial.values()))), dtype=bool)
        lower, upper = self.grid.spatial_bounds()
        for values in spatial.values():
            mask |= np.isclose(values[:, 0], lower) | np.isclose(values[:, 0], upper)
        return mask

    def config(
        self, device: str, steps: int, seed: int, evaluating: bool = False
    ) -> dict[str, Any]:
        domain = self.grid.domain(self.dimensions, evaluating)
        return {
            "version": 1,
            "domain": domain,
            "network": self.budget.network(),
            "training": self.budget.training(steps, seed, evaluating),
            "validation": False,
            "boundaries": self.boundaries(tuple(domain["space"])),
            "runtime": {
                "device": device,
                "precision": "32-true",
                "progress": False,
                "metrics": True,
                "save_training_predictions": False,
            },
        }


class HeatCase(BenchmarkCase):
    def reference(self, spatial: dict[str, Array], time: Array) -> Array:
        field = np.ones_like(spatial["x"])
        for values in spatial.values():
            field = field * np.sin(np.pi * values)
        return cast(Array, field * np.exp(-0.1 * self.dimensions * np.pi**2 * time))


class DirichletHeatCase(HeatCase):
    def reference(self, spatial: dict[str, Array], time: Array) -> Array:
        background = 1 + 0.2 * self.dimensions * time
        for column in spatial.values():
            background = background + column**2
        return cast(Array, background + super().reference(spatial, time))

    def additional_errors(
        self, spatial: dict[str, Array], time: Array, error: Array
    ) -> dict[str, float]:
        mode = super().reference(spatial, time)
        return {"mode_relative_l2": float(np.linalg.norm(error) / np.linalg.norm(mode))}

    def boundaries(self, axes: tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            {"kind": "dirichlet", "field": "u", "axis": axis, "value_function": "wall"}
            for axis in axes
        ]


class NeumannHeatCase(BenchmarkCase):
    def reference(self, spatial: dict[str, Array], time: Array) -> Array:
        return cast(
            Array,
            1
            + spatial["x"] * time
            + np.exp(-0.1 * np.pi**2 * time) * np.cos(np.pi * spatial["x"]),
        )

    def boundaries(self, axes: tuple[str, ...]) -> list[dict[str, Any]]:
        return [{"kind": "neumann", "field": "u", "axis": "x", "value_function": "wall"}]


class RobinHeatCase(NeumannHeatCase):
    def reference(self, spatial: dict[str, Array], time: Array) -> Array:
        mu = np.pi / 2
        return cast(
            Array,
            1
            + spatial["x"] * time
            + np.exp(-0.1 * mu**2 * time) * np.cos(mu * (spatial["x"] - 0.5)),
        )

    def boundaries(self, axes: tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            {
                "kind": "robin",
                "field": "u",
                "axis": "x",
                "alpha": float(np.pi / 2),
                "beta": 1,
                "value_function": "wall",
            }
        ]


class AdvectionCase(BenchmarkCase):
    def reference(self, spatial: dict[str, Array], time: Array) -> Array:
        return cast(Array, np.sin(np.pi * (spatial["x"] - time)))

    def boundaries(self, axes: tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            {
                "kind": "dirichlet",
                "field": "u",
                "axis": "x",
                "side": "lower",
                "value_function": "inflow",
            }
        ]

    def boundary_mask(self, spatial: dict[str, Array]) -> NDArray[np.bool_]:
        return np.isclose(spatial["x"][:, 0], self.grid.spatial_bounds()[0])


class BurgersCase(BenchmarkCase):
    def reference(self, spatial: dict[str, Array], time: Array) -> Array:
        return cast(Array, np.exp(-time) * np.sin(np.pi * spatial["x"]))


CASES: dict[str, BenchmarkCase] = {
    case.name: case
    for case in (
        HeatCase("heat2d", 2),
        HeatCase("heat3d", 3, grid=CartesianGrid((8, 17))),
        DirichletHeatCase("heat_dirichlet2d", 2),
        DirichletHeatCase("heat_dirichlet3d", 3, grid=CartesianGrid((8, 17))),
        NeumannHeatCase("heat_neumann", 1),
        RobinHeatCase("heat_robin", 1),
        AdvectionCase("advection", 1),
        BurgersCase("burgers", 1),
    )
}
