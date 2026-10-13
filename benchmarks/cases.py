"""Fixed problem definitions and independent NumPy analytical references."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
PROBLEMS = Path(__file__).parent / "problems"


@dataclass(frozen=True)
class BenchmarkCase(ABC):
    name: str
    dimensions: int
    relative_l2_limit: float = 0.05
    max_error_limit: float = 0.15
    residual_rmse_limit: float = 0.1

    @property
    def source(self) -> Path:
        return PROBLEMS / self.name / "functions.py"

    @abstractmethod
    def reference(self, spatial: dict[str, Array], time: Array) -> Array:
        """Independent analytical solution, with NumPy broadcasting."""

    def boundaries(self, axes: tuple[str, ...]) -> list[dict[str, Any]]:
        return [{"kind": "dirichlet", "field": "u", "axis": axis, "value": 0} for axis in axes]

    def boundary_mask(self, spatial: dict[str, Array]) -> NDArray[np.bool_]:
        mask = np.zeros(len(next(iter(spatial.values()))), dtype=bool)
        for values in spatial.values():
            mask |= np.isclose(values[:, 0], 0) | np.isclose(values[:, 0], 1)
        return mask

    def config(
        self, device: str, steps: int, seed: int, evaluating: bool = False
    ) -> dict[str, Any]:
        space = {
            axis: {
                "interval": [0, 1],
                "points": (17 if self.dimensions == 3 else 41)
                if evaluating
                else (8 if self.dimensions == 3 else 16),
            }
            for axis in ("x", "y", "z")[: self.dimensions]
        }
        boundaries = self.boundaries(tuple(space))
        return {
            "version": 1,
            "domain": {
                "space": space,
                "time": {"interval": [0, 0.5], "points": 31 if evaluating else 11},
            },
            "network": {"hidden_layers": [32, 32, 32]},
            "training": {
                "steps": steps,
                "epochs": steps,
                "seed": seed,
                "batch_size": 128 if not evaluating else 2048,
                "optimizer": {"name": "adam", "learning_rate": 0.001},
                "collocation": {"sampling": "lhs", "points": 1024},
            },
            "validation": False,
            "boundaries": boundaries,
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
        return np.isclose(spatial["x"][:, 0], 0)


class BurgersCase(BenchmarkCase):
    def reference(self, spatial: dict[str, Array], time: Array) -> Array:
        return cast(Array, np.exp(-time) * np.sin(np.pi * spatial["x"]))


CASES: dict[str, BenchmarkCase] = {
    case.name: case
    for case in (
        HeatCase("heat2d", 2),
        HeatCase("heat3d", 3),
        AdvectionCase("advection", 1),
        BurgersCase("burgers", 1),
    )
}
