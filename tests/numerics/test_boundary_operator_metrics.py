"""Face temperature and operator errors have distinct physical meanings."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from benchmarks.boundary_metrics import boundary_operator_errors
from benchmarks.cases import CASES
from benchmarks.metrics import solution_errors
from pinnstorch.api import Coordinates, TensorFields
from pinnstorch.runner.problem import ProblemAdapter, load_problem
from pinnstorch.runner.settings import TaskSettings


@pytest.mark.parametrize("kind", ["dirichlet", "neumann", "robin", "heat_flux"])
def test_boundary_metric_checks_normals_and_each_operator(kind: str) -> None:
    case = CASES["heat_neumann_polynomial2d"]
    config = case.config("cpu", 2, 19)
    faces = []
    for side, normal, value in (("lower", -1, 1), ("upper", 1, 2)):
        boundary = dict(kind=kind, axis="x", field="u", side=side, value=value)
        if kind == "neumann":
            boundary["value"] = normal
        elif kind == "robin":
            boundary.update(alpha=2, beta=3, value=2 * value + 3 * normal)
        elif kind == "heat_flux":
            boundary.update(conductivity=0.4, value=-0.4 * normal)
        faces.append(boundary)
    config["boundaries"] = faces
    settings = TaskSettings.read(config)
    problem = load_problem(case.source)
    adapter = ProblemAdapter(problem, settings.domain)
    observed: list[torch.Tensor] = []

    def field(spatial: list[torch.Tensor], clock: torch.Tensor) -> TensorFields:
        observed.append(torch.cat((*spatial, clock), dim=1).detach())
        return {"u": spatial[0] + 1}

    rows = boundary_operator_errors(settings, adapter, field, "cpu", 13, dtype=torch.float64)
    assert len(rows) == 2
    assert max(row.max_error for row in rows) < 1e-14
    assert all(row.points == 13 and row.kind == kind for row in rows)
    assert all(torch.all((x[:, 1] > 0) & (x[:, 1] < 1)) for x in observed)
    assert all(x[0, -1] == 0 and x[1, -1] == 0.5 for x in observed)
    before = [x.clone() for x in observed]
    observed.clear()
    boundary_operator_errors(settings, adapter, field, "cpu", 13, dtype=torch.float64)
    assert all(torch.equal(x, y) for x, y in zip(before, observed))


def test_neumann_detects_slope_but_accepts_constant_temperature_error() -> None:
    case = CASES["heat_neumann_polynomial2d"]
    settings = TaskSettings.read(case.config("cpu", 2, 19))
    problem = load_problem(case.source)
    adapter = ProblemAdapter(problem, settings.domain)

    def shifted(spatial: list[torch.Tensor], clock: torch.Tensor) -> TensorFields:
        return {"u": problem.initial(Coordinates(dict(zip("xy", spatial)), clock))["u"] + 3}

    rows = boundary_operator_errors(settings, adapter, shifted, "cpu", dtype=torch.float64)
    assert max(row.max_error for row in rows) < 1e-14

    def tilted(spatial: list[torch.Tensor], clock: torch.Tensor) -> TensorFields:
        return {"u": shifted(spatial, clock)["u"] + 0.25 * spatial[0]}

    rows = boundary_operator_errors(settings, adapter, tilted, "cpu", dtype=torch.float64)
    assert all(row.rmse == pytest.approx(0.25) for row in rows if row.axis == "x")
    assert all(row.rmse < 1e-14 for row in rows if row.axis == "y")
    spatial = {axis: np.array([[0], [0.5], [1]], dtype=float) for axis in "xy"}
    clock = np.array([[0.0], [0.5]])
    metrics = solution_errors(case, spatial, clock, case.reference(spatial, clock.T) + 3)
    assert metrics["boundary_temperature_rmse"] == metrics["boundary_rmse"] == 3


@pytest.mark.parametrize("conductivity", ["k", "coefficient"])
def test_heat_flux_supports_parameter_or_coefficient(conductivity: str) -> None:
    case = CASES["heat_neumann_polynomial1d"]
    config = case.config("cpu", 2, 19)
    config["domain"]["parameters"] = {"k": {"interval": [0.1, 0.5], "points": 3}}
    config["boundaries"] = [
        dict(
            kind="heat_flux",
            field="u",
            axis="x",
            side="upper",
            conductivity=conductivity,
            value_function="flux",
        )
    ]
    original = load_problem(case.source)

    def flux(c: Coordinates) -> torch.Tensor:
        return -2 * c.parameters["k"]

    problem = replace(
        original,
        boundary_values={"flux": flux},
        coefficients={"coefficient": lambda c: c.parameters["k"]},
    )
    settings = TaskSettings.read(config)
    adapter = ProblemAdapter(problem, settings.domain)

    def field(spatial: list[torch.Tensor], clock: torch.Tensor) -> TensorFields:
        return {"u": 2 * spatial[0]}

    rows = boundary_operator_errors(settings, adapter, field, "cpu", dtype=torch.float64)
    assert rows[0].max_error < 1e-14


def test_boundary_metric_rejects_malformed_or_nonfinite_data() -> None:
    case = CASES["heat_neumann_polynomial1d"]
    config = case.config("cpu", 2, 19)
    config["boundaries"][0] = dict(kind="dirichlet", axis="x", field="u", value_function="bad")
    settings = TaskSettings.read(config)
    original = load_problem(case.source)

    def field(spatial: list[torch.Tensor], clock: torch.Tensor) -> TensorFields:
        return {"u": spatial[0]}

    for callback in (
        lambda c: torch.full_like(c.t, float("nan")),
        lambda c: torch.ones((len(c.t), 2)),
    ):
        adapter = ProblemAdapter(
            replace(original, boundary_values={"bad": callback}), settings.domain
        )
        with pytest.raises(ValueError):
            boundary_operator_errors(settings, adapter, field, "cpu")


def test_face_sampling_uses_shifted_physical_bounds_and_parameter_range() -> None:
    case = CASES["heat_neumann_polynomial2d"]
    config = case.config("cpu", 2, 19)
    config["domain"]["space"]["x"]["interval"] = [-2, 3]
    config["domain"]["space"]["y"]["interval"] = [4, 7]
    config["domain"]["parameters"] = {"k": {"interval": [0.1, 0.5], "points": 3}}
    config["domain"]["time"]["interval"] = [2, 4]
    config["boundaries"] = [dict(kind="dirichlet", field="u", axis="x", side="lower", value=0)]
    settings = TaskSettings.read(config)
    adapter = ProblemAdapter(load_problem(case.source), settings.domain)

    def field(spatial: list[torch.Tensor], clock: torch.Tensor) -> TensorFields:
        assert len(spatial) == 3
        assert torch.all(spatial[0] == -2)
        assert torch.all((spatial[1] > 4) & (spatial[1] < 7))
        assert torch.all((spatial[2] >= 0.1) & (spatial[2] <= 0.5))
        assert clock[0] == 2 and clock[1] == 4
        return {"u": torch.zeros_like(clock)}

    assert boundary_operator_errors(settings, adapter, field, "cpu")[0].rmse == 0
    with pytest.raises(ValueError, match="At least two"):
        boundary_operator_errors(settings, adapter, field, "cpu", 1)
    config["domain"]["space"]["y"]["interval"] = [1, 1 + np.finfo(np.float32).eps]
    tiny = TaskSettings.read(config)
    with pytest.raises(ValueError, match="not representable"):
        boundary_operator_errors(
            tiny, ProblemAdapter(load_problem(case.source), tiny.domain), field, "cpu"
        )


def test_neumann_metric_rejects_nan_field_instead_of_differentiating_it_to_zero() -> None:
    case = CASES["heat_neumann_polynomial1d"]
    settings = TaskSettings.read(case.config("cpu", 2, 19))
    adapter = ProblemAdapter(load_problem(case.source), settings.domain)

    def field(spatial: list[torch.Tensor], clock: torch.Tensor) -> TensorFields:
        return {"u": torch.full_like(clock, float("nan"))}

    with pytest.raises(ValueError, match="physical field"):
        boundary_operator_errors(settings, adapter, field, "cpu")
