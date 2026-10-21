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
    "burgers_unforced",
    "conductivity",
    "constant_surrogate",
    "spatial_scale_surrogate",
    "quadratic_shapes",
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
    space_interval: tuple[float, float] = (0, 1)
    time_interval: tuple[float, float] | None = None
    time_counts: tuple[int, int] = (11, 31)

    @property
    def convergence(self) -> bool:
        return self.family not in (
            "advection",
            "burgers",
            "burgers_unforced",
            "constant_surrogate",
            "spatial_scale_surrogate",
            "quadratic_shapes",
        )

    @property
    def modal_metric(self) -> bool:
        return self.family in ("dirichlet", "neumann_box", "robin_box", "mixed", "homogeneous")

    @property
    def static_background(self) -> bool:
        return self.family in ("homogeneous", "polynomial")

    def faces(self) -> list[dict[str, str]]:
        result = []
        for axis in ("x", "y", "z")[: self.dimensions]:
            if self.family in (
                "heat",
                "dirichlet",
                "burgers",
                "advection",
                "burgers_unforced",
                "constant_surrogate",
                "spatial_scale_surrogate",
                "quadratic_shapes",
            ):
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


CONSTANT_EVALUATION = dict(
    training_interval=[0.05, 0.25],
    training_sampling="joint_continuous_lhs",
    hypotheses=[0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25],
    hypothesis_scope="interpolation_slices_not_unseen_distribution",
    grid=[101, 51],
    pde_points=1024,
    pde_seed=20261021,
    ignored_k=0.15,
    field_norm="array_l2_per_hypothesis",
    aggregation="worst_hypothesis",
)

SPATIAL_SCALE_EVALUATION = {
    **{key: value for key, value in CONSTANT_EVALUATION.items() if key != "ignored_k"},
    "parameter": "a",
    "ignored_a": 0.15,
    "coefficient_profile": "a*(1+2*x*(1-x))",
    "initial_field": "4*x*(1-x)",
}


def _binary_reverse(n: int) -> float:
    value, weight = 0.0, 0.5
    while n:
        value += (n % 2) * weight
        n //= 2
        weight /= 2
    return value


QUADRATIC_TRAINING_ROWS = [
    *[[a, b] for a in (0.05, 0.15) for b in (1.0, 2.0)],
    *[[0.05 + 0.1 * (i + 0.5) / 12, 1 + _binary_reverse(i + 1)] for i in range(12)],
]
QUADRATIC_UNSEEN_ROWS = [[a, b] for a in (0.075, 0.125) for b in (1.2, 1.4, 1.6, 1.8)]
QUADRATIC_EVALUATION = dict(
    parameters=["a", "b"],
    training_rows=QUADRATIC_TRAINING_ROWS,
    hypotheses=QUADRATIC_UNSEEN_ROWS + QUADRATIC_TRAINING_ROWS[:4],
    splits=["unseen"] * 8 + ["training_corner"] * 4,
    training_sampling="physical_lhs_complete_code_rows",
    coefficient_profile="a*(1+x/b)**2",
    initial_field="sin(pi*x)**3",
    grid=[101, 101],
    reference_modes=64,
    pde_points=1024,
    pde_seed=20261021,
    ignored_a=0.1,
    ignored_b=1.5,
    response_pairs=[[0, 3], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]],
    field_norm="array_l2_per_hypothesis",
    aggregation="worst_hypothesis",
)
QUADRATIC_METRICS = {"dx_relative_l2", "flux_relative_l2", "coefficient_response_relative_l2"}


SURROGATE_EVALUATIONS = {
    "heat_quadratic_shapes": QUADRATIC_EVALUATION,
    "heat_constant_surrogate": CONSTANT_EVALUATION,
    "heat_spatial_scale_surrogate": SPATIAL_SCALE_EVALUATION,
}

UNFORCED_EVALUATION = dict(
    coarse_grid=[801, 201],
    fine_grid=[1601, 401],
    global_points=2048,
    global_seed=20261013,
    regional_points=2048,
    regional_seed=20261014,
    regional_space=[-0.1, 0.1],
    regional_time=[0.15, 1.0],
    field_norm="array_l2",
    gradient_norm="spatial_trapezoid_worst_time",
    refinement_norm="difference_over_registered_limit",
    reference_order=512,
)
UNFORCED_METRICS = {
    "regional_pde_rmse",
    "worst_time_gradient_relative_l2",
    "grid_refinement_ratio",
}
UNFORCED_VISCOSITIES = {"burgers_unforced_nu010": 0.1, "burgers_unforced_nu001": 0.01}

SPECS = (
    CaseSpec(
        "heat_quadratic_shapes",
        "quadratic_shapes",
        1,
        (16, 101),
        time_interval=(0, 1),
        time_counts=(11, 101),
    ),
    CaseSpec(
        "heat_spatial_scale_surrogate",
        "spatial_scale_surrogate",
        1,
        (16, 101),
        time_interval=(0, 1),
        time_counts=(11, 51),
    ),
    CaseSpec(
        "heat_constant_surrogate",
        "constant_surrogate",
        1,
        (16, 101),
        time_interval=(0, 1),
        time_counts=(11, 51),
    ),
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
    *[
        CaseSpec(
            name,
            "burgers_unforced",
            1,
            (16, 801),
            space_interval=(-1, 1),
            time_interval=(0, 1),
            time_counts=(11, 201),
        )
        for name in UNFORCED_VISCOSITIES
    ],
)
CATALOG = {spec.name: spec for spec in SPECS}
CONVERGENCE_CASES = tuple(spec.name for spec in SPECS if spec.convergence)
