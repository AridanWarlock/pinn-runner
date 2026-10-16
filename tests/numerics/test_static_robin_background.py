"""Initial-only backgrounds preserve physical derivatives on shifted boxes."""

import os
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import torch

from benchmarks.cases import CASES
from pinnstorch.api import Coordinates, Equation, TensorFields
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings

BOUNDS = ((-2.0, 0.5), (0.2, 1.7), (1.0, 3.4))


def initial_polynomial(c: Coordinates) -> TensorFields:
    value = torch.ones_like(c.t)
    for column, (lower, upper) in zip(c.spatial.values(), BOUNDS):
        x = (column - lower) / (upper - lower)
        value = value * (3 * x.square() - 2 * x.pow(3))
    return {"u": value}


def initial_affine(c: Coordinates) -> TensorFields:
    value = torch.ones_like(c.t)
    for x in c.spatial.values():
        value = value + x
    return {"u": value}


def initial_constant(c: Coordinates) -> TensorFields:
    return {"u": torch.ones_like(c.t)}


def recipe(dimensions: int, condition: str, execution: str) -> dict[str, Any]:
    config = CASES[f"heat_neumann_polynomial{dimensions}d"].config(
        os.environ.get("PINN_TEST_DEVICE", "cpu"), 2, 19
    )
    config["network"]["hidden_layers"] = [4, 4]
    for axis, bounds in zip(config["domain"]["space"].values(), BOUNDS):
        axis.update(interval=list(bounds), points=3)
    config["domain"]["time"].update(interval=[0.7, 1.9], points=3)
    config["training"].update(constraints="hard_initial_robin", hard_execution=execution)
    if condition == "flux":
        config["boundaries"] = [
            dict(kind="neumann", field="u", axis=axis, side=side, value=value)
            for axis in "xyz"[:dimensions]
            for side, value in (("lower", -1), ("upper", 1))
        ]
    elif condition == "robin":
        config["boundaries"] = [
            dict(kind="robin", field="u", axis=axis, alpha=2, beta=1, value=2)
            for axis in "xyz"[:dimensions]
        ]
    return config


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("condition", ["zero", "flux", "robin"])
@pytest.mark.parametrize("execution", ["eager", "robin_grouped"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_static_background_preserves_field_derivatives_and_gradients(
    dimensions: int,
    condition: str,
    execution: str,
    dtype: torch.dtype,
    tmp_path: Path,
    lifting: str = "recursive",
) -> None:
    case = CASES[f"heat_neumann_polynomial{dimensions}d"]
    initial = {"zero": initial_polynomial, "flux": initial_affine, "robin": initial_constant}[
        condition
    ]
    problem = replace(load_problem(case.source), initial=initial)
    config = recipe(dimensions, condition, execution)
    config["training"]["hard_lifting"] = lifting
    positions = torch.tensor(
        [
            *product((0.0, 1.0), repeat=dimensions),
            [0.4] * dimensions,
            [1e-5] * dimensions,
            [1 - 1e-5] * dimensions,
        ],
        dtype=dtype,
        device=config["runtime"]["device"],
    )
    results = []
    signatures = []
    reference: dict[str, torch.Tensor] | None = None
    for mode in ("general", "static", "auto"):
        config["training"]["hard_background"] = mode
        factory = SolverFactory(TaskSettings.read(config), problem, tmp_path, case.source.parent)
        model = factory.build(True).model.to(device=positions.device, dtype=dtype)
        if reference is None:
            reference = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            model.load_state_dict(reference)
        assert model.hparams["hard_background"] == ("general" if mode == "general" else "static")
        signatures.append(factory.constraints.signature)
        columns = [
            (positions[:, i : i + 1] * (upper - lower) + lower).detach().requires_grad_()
            for i, (lower, upper) in enumerate(BOUNDS[:dimensions])
        ]
        clock = (
            torch.linspace(0.7, 1.9, len(positions), device=positions.device, dtype=dtype)
            .reshape(-1, 1)
            .requires_grad_()
        )
        physical = dict(model(columns, clock))
        c = Coordinates(dict(zip("xyz", columns)), clock)
        e = Equation(physical, c)
        values = [physical["u"], e.dt("u"), e.laplacian("u")]
        values.extend(e.diff("u", axis) for axis in c.spatial)
        residual = factory.adapter.residuals(physical, *columns, clock)["heat"]
        gradients = torch.autograd.grad(residual.square().mean(), tuple(model.parameters()))
        results.append([v.detach().clone() for v in (*values, residual, *gradients)])
        origin = torch.full_like(clock, 0.7)
        torch.testing.assert_close(
            model(columns, origin)["u"],
            initial(Coordinates(c.spatial, origin))["u"],
            rtol=0,
            atol=0,
        )
    assert len(set(signatures)) == 1
    tolerance = 3e-5 if dtype == torch.float32 else 1e-11
    for alternative in results[1:]:
        for expected, actual in zip(results[0], alternative):
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_background_guards_callbacks_and_keeps_initial_validation(tmp_path: Path) -> None:
    case = CASES["heat_neumann2d"]
    problem = load_problem(case.source)
    config = case.config("cpu", 2, 19)
    config["training"].update(constraints="hard_initial_robin", hard_background="static")
    with pytest.raises(ValueError, match="explicit numeric"):
        SolverFactory(TaskSettings.read(config), problem, tmp_path, case.source.parent)
    config["training"]["hard_background"] = "auto"
    factory = SolverFactory(TaskSettings.read(config), problem, tmp_path, case.source.parent)
    assert factory.build(True).model.hparams["hard_background"] == "general"
    # A constant-valued callback still requires the general path; do not infer metadata.
    constant = replace(
        problem,
        boundary_values={
            name: lambda c: torch.zeros_like(c.t) for name in problem.boundary_values
        },
    )
    factory = SolverFactory(TaskSettings.read(config), constant, tmp_path, case.source.parent)
    assert factory.build(True).model.hparams["hard_background"] == "general"
    config = recipe(2, "zero", "robin_grouped")
    config["training"]["hard_background"] = "static"
    config["boundaries"][0]["value"] = 1
    polynomial = replace(
        load_problem(CASES["heat_neumann_polynomial2d"].source), initial=initial_polynomial
    )
    with pytest.raises(ValueError, match="initial Robin/Neumann"):
        SolverFactory(TaskSettings.read(config), polynomial, tmp_path, case.source.parent)


@pytest.mark.parametrize("constraints", ["soft", "hard_initial_dirichlet"])
def test_background_rejected_without_robin_constraints(constraints: str) -> None:
    config = CASES["heat2d"].config("cpu", 2, 19)
    config["training"].update(constraints=constraints, hard_background="static")
    with pytest.raises(ValueError, match="hard_background"):
        TaskSettings.read(config)


@pytest.mark.parametrize(
    ("case_name", "constraints", "expected"),
    [
        ("heat_neumann_polynomial1d", "hard_initial_robin", "static"),
        ("heat_neumann_polynomial2d", "hard_initial_robin", "static"),
        ("heat_neumann_polynomial3d", "hard_initial_robin", "static"),
        ("heat_neumann2d", "hard_initial_robin", "general"),
        ("heat2d", "soft", "general"),
        ("heat2d", "hard_initial_dirichlet", "general"),
    ],
)
def test_default_background_selects_only_eligible_robin_data(
    case_name: str, constraints: str, expected: str, tmp_path: Path
) -> None:
    case = CASES[case_name]
    config = case.config("cpu", 2, 19)
    config["training"]["constraints"] = constraints
    assert "hard_background" not in config["training"]
    settings = TaskSettings.read(config)
    problem = load_problem(case.source)
    factory = SolverFactory(settings, problem, tmp_path, case.source.parent)
    assert factory.build(True).model.hparams["hard_background"] == expected
    config["training"]["hard_background"] = "general"
    control = SolverFactory(TaskSettings.read(config), problem, tmp_path, case.source.parent)
    assert control.build(True).model.hparams["hard_background"] == "general"


def test_static_convergence_keeps_independent_general_evaluator(tmp_path: Path) -> None:
    from benchmarks.convergence import ConvergenceArm, ConvergencePlan, convergence_trial
    from benchmarks.policies import CartesianGrid, TrainingBudget

    case = replace(
        CASES["heat_neumann_polynomial2d"],
        grid=CartesianGrid((3, 3), (3, 3)),
        budget=TrainingBudget(layers=(4, 4), collocation_points=7),
    )
    plan = ConvergencePlan(case, 3, (0, 1, 2), {"relative_l2": 1}, "hard_initial_robin", 7, "cpu")
    trials = [
        convergence_trial(
            plan, ConvergenceArm("robin_grouped", "recursive", "all", mode), 19, tmp_path / mode
        )
        for mode in ("general", "static")
    ]
    assert trials[1]["resolved_background"] == "static"
    assert trials[1]["control_prediction_background"] == "general"
    assert trials[1]["control_prediction_execution"] == "eager"
    assert trials[0]["initial_state_sha256"] == trials[1]["initial_state_sha256"]
    assert trials[0]["fixed_objective_sha256"] == trials[1]["fixed_objective_sha256"]
    for left, right in zip(trials[0]["snapshots"], trials[1]["snapshots"]):
        assert left["metrics"] == right["metrics"]
        assert left["closures"] == right["closures"]


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_static_background_with_expanded_lifting(dimensions: int, tmp_path: Path) -> None:
    test_static_background_preserves_field_derivatives_and_gradients(
        dimensions, "zero", "robin_grouped", torch.float64, tmp_path, "expanded"
    )


@pytest.mark.parametrize("dimensions", [2, 3])
def test_static_background_directional_weight_derivative(dimensions: int, tmp_path: Path) -> None:
    case = CASES[f"heat_neumann_polynomial{dimensions}d"]
    base = case.config("cpu", 2, 19)
    base["network"]["hidden_layers"] = [8, 8]
    base["training"].update(constraints="hard_initial_robin", hard_execution="robin_grouped")
    problem = load_problem(case.source)
    generator = torch.Generator().manual_seed(73)
    points = torch.rand((13, dimensions + 1), dtype=torch.float64, generator=generator)
    points[:, -1] *= 0.5
    state = None
    measured = []
    for background in ("general", "static"):
        base["training"]["hard_background"] = background
        factory = SolverFactory(TaskSettings.read(base), problem, tmp_path, case.source.parent)
        model = factory.build(True).model.double()
        if state is None:
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            model.load_state_dict(state)
        parameters = tuple(model.parameters())
        direction_generator = torch.Generator().manual_seed(73)
        directions = [
            torch.randn(p.shape, dtype=p.dtype, generator=direction_generator) for p in parameters
        ]
        norm = torch.cat([d.flatten() for d in directions]).norm()
        directions = [d / norm for d in directions]

        def loss() -> torch.Tensor:
            columns = [
                points[:, i : i + 1].detach().requires_grad_() for i in range(dimensions + 1)
            ]
            output = dict(model(columns[:-1], columns[-1]))
            return factory.adapter.residuals(output, *columns)["heat"].square().mean()

        value = loss()
        grads = torch.autograd.grad(value, parameters)
        derivative = sum((g * d).sum() for g, d in zip(grads, directions))
        epsilon = 1e-5
        with torch.no_grad():
            for p, d in zip(parameters, directions):
                p.add_(d, alpha=epsilon)
        plus = loss().detach()
        with torch.no_grad():
            for p, d in zip(parameters, directions):
                p.add_(d, alpha=-2 * epsilon)
        minus = loss().detach()
        finite_difference = (plus - minus) / (2 * epsilon)
        torch.testing.assert_close(derivative, finite_difference, atol=1e-9, rtol=2e-6)
        measured.append((value.detach(), derivative.detach()))
    for before, after in zip(measured[0], measured[1]):
        torch.testing.assert_close(before, after, atol=1e-12, rtol=1e-12)
