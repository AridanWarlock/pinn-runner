"""Fixed problem definitions and independent NumPy analytical references."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .catalog import SPECS, Family
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

    def prepare_source(self, directory: Path, representation: str | None = None) -> Path:
        if representation is not None:
            raise ValueError("Coefficient representation requires a conductivity case")
        return self.source

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


@dataclass(frozen=True)
class HeatAxisMode:
    frequency: float
    shift: float
    alpha: float


class BoxNeumannHeatCase(BenchmarkCase):
    @property
    def affine_background_rate(self) -> float:
        return 1.0

    def mode_for(self, axis: str) -> HeatAxisMode:
        return HeatAxisMode(float(np.pi), 0, 0)

    def mode(self, spatial: dict[str, Array], time: Array) -> Array:
        field = np.ones_like(spatial["x"])
        decay = 0.0
        for axis, column in spatial.items():
            mode = self.mode_for(axis)
            field = field * np.cos(mode.frequency * (column - mode.shift))
            decay += mode.frequency**2
        return cast(Array, field * np.exp(-0.1 * decay * time))

    def reference(self, spatial: dict[str, Array], time: Array) -> Array:
        background = np.ones_like(spatial["x"]) * np.ones_like(time)
        for column in spatial.values():
            background = background + self.affine_background_rate * time * column
        return cast(Array, background + self.mode(spatial, time))

    def additional_errors(
        self, spatial: dict[str, Array], time: Array, error: Array
    ) -> dict[str, float]:
        mode = self.mode(spatial, time)
        norm_squared = float(np.sum(mode**2))
        if norm_squared == 0:
            raise ValueError("Modal diagnostics require a nonzero mode")
        coefficient_error = float(np.sum(error * mode) / norm_squared)
        orthogonal = error - coefficient_error * mode
        return {
            "mode_relative_l2": float(np.linalg.norm(error) / np.sqrt(norm_squared)),
            "mode_projection_relative_error": abs(coefficient_error),
            "mode_orthogonal_rmse": float(np.sqrt(np.mean(orthogonal**2))),
        }

    def boundaries(self, axes: tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            dict(
                kind="neumann" if mode.alpha == 0 else "robin",
                field="u",
                axis=axis,
                value_function=f"wall_{axis}",
                **({"alpha": mode.alpha, "beta": 1} if mode.alpha else {}),
            )
            for axis in axes
            for mode in (self.mode_for(axis),)
        ]


class BoxRobinHeatCase(BoxNeumannHeatCase):
    def mode_for(self, axis: str) -> HeatAxisMode:
        return HeatAxisMode(float(np.pi / 2), 0.5, float(np.pi / 2))


class HomogeneousNeumannHeatCase(BoxNeumannHeatCase):
    @property
    def affine_background_rate(self) -> float:
        return 0.0

    @property
    def source(self) -> Path:
        return PROBLEMS / "heat_neumann_homogeneous" / "functions.py"

    def boundaries(self, axes: tuple[str, ...]) -> list[dict[str, Any]]:
        return [{"kind": "neumann", "field": "u", "axis": axis, "value": 0} for axis in axes]


class PolynomialNeumannHeatCase(BenchmarkCase):
    """Manufactured smooth solution with nonzero third boundary derivatives."""

    @property
    def source(self) -> Path:
        return PROBLEMS / "heat_neumann_polynomial" / "functions.py"

    def reference(self, spatial: dict[str, Array], time: Array) -> Array:
        field = np.ones_like(time)
        for column in spatial.values():
            field = field * (3 * column**2 - 2 * column**3)
        return cast(Array, np.exp(-time) * field)

    def boundaries(self, axes: tuple[str, ...]) -> list[dict[str, Any]]:
        return [dict(kind="neumann", field="u", axis=axis, value=0) for axis in axes]


class ConductivityHeatCase(PolynomialNeumannHeatCase):
    def additional_errors(
        self, spatial: dict[str, Array], time: Array, error: Array
    ) -> dict[str, float]:
        x, y = (np.unique(spatial[axis]) for axis in ("x", "y"))
        order = np.lexsort((spatial["y"].reshape(-1), spatial["x"].reshape(-1)))
        if len(x) * len(y) != len(error):
            raise ValueError("Mean heat requires a Cartesian evaluation grid")
        volume_error = error[order].reshape(len(x), len(y), -1)
        integral = np.trapz(np.trapz(volume_error, y, axis=1), x, axis=0)
        return {"mean_heat_max_error": float(np.max(np.abs(integral)))}

    @property
    def source(self) -> Path:
        raise ValueError("Conductivity source requires explicit formula/table preparation")

    def prepare_source(self, directory: Path, representation: str | None = None) -> Path:
        from .conductivity import materialized_source

        if representation not in ("formula", "table"):
            raise ValueError("Conductivity case requires formula/table representation")
        path = directory / "functions.py"
        path.write_text(materialized_source(self.name, representation), encoding="utf-8")
        return path


class MixedBoxHeatCase(BoxNeumannHeatCase):
    def mode_for(self, axis: str) -> HeatAxisMode:
        return (
            HeatAxisMode(float(np.pi / 2), 0.5, float(np.pi / 2))
            if axis == "y"
            else super().mode_for(axis)
        )


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


FACTORIES: dict[Family, type[BenchmarkCase]] = {
    "heat": HeatCase,
    "dirichlet": DirichletHeatCase,
    "neumann_interval": NeumannHeatCase,
    "robin_interval": RobinHeatCase,
    "neumann_box": BoxNeumannHeatCase,
    "robin_box": BoxRobinHeatCase,
    "mixed": MixedBoxHeatCase,
    "polynomial": PolynomialNeumannHeatCase,
    "homogeneous": HomogeneousNeumannHeatCase,
    "conductivity": ConductivityHeatCase,
    "advection": AdvectionCase,
    "burgers": BurgersCase,
}
CASES: dict[str, BenchmarkCase] = {
    spec.name: FACTORIES[spec.family](
        spec.name,
        spec.dimensions,
        grid=CartesianGrid(
            spec.grid_counts or (16, 41),
            time_interval=(0, 1) if spec.family == "conductivity" else (0, 0.5),
        ),
        budget=TrainingBudget(layers=spec.layers),
    )
    for spec in SPECS
}
