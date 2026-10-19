"""Case identity and executor capabilities without numerical-library imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Family = Literal[
    "heat",
    "dirichlet",
    "neumann_interval",
    "robin_interval",
    "neumann_box",
    "robin_box",
    "mixed",
    "polynomial",
    "homogeneous",
    "advection",
    "burgers",
    "conductivity",
]
CONSTRAINTS = ("soft", "hard_initial_dirichlet", "hard_initial_robin")
EXECUTIONS = ("eager", "product_rule", "robin_batch", "robin_grouped", "neumann_analytic")
LIFTINGS = ("recursive", "expanded")
BACKWARDS = ("auto", "all", "parameters")
BACKGROUNDS = ("general", "static", "auto")
ENCODINGS = ("identity", "neumann_cosine", "neumann_polynomial")


@dataclass(frozen=True)
class CaseSpec:
    name: str
    family: Family
    dimensions: int
    grid_counts: tuple[int, int] | None = None
    layers: tuple[int, ...] = (32, 32, 32)

    @property
    def convergence(self) -> bool:
        return self.family not in ("advection", "burgers")

    @property
    def modal_metric(self) -> bool:
        return self.family in ("dirichlet", "neumann_box", "robin_box", "mixed", "homogeneous")

    @property
    def static_background(self) -> bool:
        return self.family in ("homogeneous", "polynomial")

    def faces(self) -> list[dict[str, str]]:
        result = []
        for axis in ("x", "y", "z")[: self.dimensions]:
            if self.family in ("heat", "dirichlet", "burgers", "advection"):
                kind = "dirichlet"
            elif self.family in ("robin_interval", "robin_box") or (
                self.family == "mixed" and axis == "y"
            ):
                kind = "robin"
            else:
                kind = "neumann"
            for side in ("lower",) if self.family == "advection" else ("lower", "upper"):
                result.append(dict(field="u", kind=kind, axis=axis, side=side))
        return result


SPECS = (
    CaseSpec("heat2d", "heat", 2),
    CaseSpec("heat3d", "heat", 3, (8, 17)),
    CaseSpec("heat_dirichlet2d", "dirichlet", 2),
    CaseSpec("heat_dirichlet3d", "dirichlet", 3, (8, 17)),
    CaseSpec("heat_neumann", "neumann_interval", 1),
    CaseSpec("heat_robin", "robin_interval", 1),
    CaseSpec("heat_neumann2d", "neumann_box", 2, (8, 17)),
    CaseSpec("heat_neumann3d", "neumann_box", 3, (6, 9)),
    CaseSpec("heat_robin2d", "robin_box", 2, (8, 17)),
    CaseSpec("heat_robin3d", "robin_box", 3, (6, 9)),
    CaseSpec("heat_mixed3d", "mixed", 3, (6, 9)),
    CaseSpec("heat_neumann_polynomial1d", "polynomial", 1),
    CaseSpec("heat_neumann_homogeneous1d", "homogeneous", 1),
    CaseSpec("heat_neumann_homogeneous2d", "homogeneous", 2, (8, 17)),
    CaseSpec("heat_neumann_homogeneous3d", "homogeneous", 3, (6, 9)),
    CaseSpec("heat_neumann_polynomial2d", "polynomial", 2, (8, 17)),
    CaseSpec("heat_neumann_polynomial3d", "polynomial", 3, (6, 9)),
    CaseSpec("heat_conductivity_beta0", "conductivity", 2, (8, 41)),
    CaseSpec("heat_conductivity_beta1", "conductivity", 2, (8, 41)),
    CaseSpec("advection", "advection", 1),
    CaseSpec("burgers", "burgers", 1),
)
CATALOG = {spec.name: spec for spec in SPECS}
CONVERGENCE_CASES = tuple(spec.name for spec in SPECS if spec.convergence)
