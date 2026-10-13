"""Fixed problem definitions and independent NumPy analytical references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
PROBLEMS = Path(__file__).parent / "problems"


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    dimensions: int
    relative_l2_limit: float = 0.05
    max_error_limit: float = 0.15
    residual_rmse_limit: float = 0.1

    @property
    def source(self) -> Path:
        return PROBLEMS / self.name / "functions.py"

    def reference(self, spatial: dict[str, Array], time: Array) -> Array:
        x = spatial["x"]
        if self.name == "heat2d":
            return cast(
                Array,
                np.sin(np.pi * x) * np.sin(np.pi * spatial["y"]) * np.exp(-0.2 * np.pi**2 * time),
            )
        if self.name == "advection":
            return cast(Array, np.sin(np.pi * (x - time)))
        if self.name == "burgers":
            return cast(Array, np.exp(-time) * np.sin(np.pi * x))
        raise ValueError(f"Unknown benchmark {self.name}")

    def config(
        self, device: str, steps: int, seed: int, evaluating: bool = False
    ) -> dict[str, Any]:
        space = {
            axis: {"interval": [0, 1], "points": 41 if evaluating else 16}
            for axis in ("x", "y", "z")[: self.dimensions]
        }
        boundaries: list[dict[str, Any]] = [
            {"kind": "dirichlet", "field": "u", "axis": axis, "value": 0} for axis in space
        ]
        if self.name == "advection":
            boundaries = [
                {
                    "kind": "dirichlet",
                    "field": "u",
                    "axis": "x",
                    "side": "lower",
                    "value_function": "inflow",
                }
            ]
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


CASES = {
    case.name: case
    for case in (
        BenchmarkCase("heat2d", 2),
        BenchmarkCase("advection", 1),
        BenchmarkCase("burgers", 1),
    )
}
