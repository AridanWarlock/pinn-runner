"""Physical chain rule and exact initial/Neumann data for experimental representations."""

import os
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import torch

from benchmarks.cases import CASES
from pinnstorch.api import Coordinates, Equation, TensorFields
from pinnstorch.models.net.spatial_features import NeumannFeatures
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings

BOUNDS = ((-2.0, 0.5), (0.2, 1.7), (1.0, 3.4))


def physical_points(
    dimensions: int, dtype: torch.dtype
) -> tuple[list[torch.Tensor], torch.Tensor]:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    unit = torch.tensor(
        [*product((0.0, 1.0), repeat=dimensions), [0.3] * dimensions, [0.71] * dimensions],
        dtype=dtype,
        device=device,
    )
    columns = [
        (unit[:, i : i + 1] * (b - a) + a).detach().requires_grad_()
        for i, (a, b) in enumerate(BOUNDS[:dimensions])
    ]
    clock = torch.full_like(columns[0], 1.3, requires_grad=True)
    return columns, clock


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_polynomial_feature_relation_vanishes_with_physical_derivatives(
    dimensions: int, dtype: torch.dtype
) -> None:
    columns, clock = physical_points(dimensions, dtype)
    features = NeumannFeatures(BOUNDS[:dimensions], "neumann_polynomial")(columns)
    value = torch.zeros_like(clock)
    for q, scaled_p in zip(features[::2], features[1::2]):
        p = scaled_p / 16
        value = value + (q * (1 - q) - 3 * p).square() - 16 * p.pow(3)
    equation = Equation({"u": value}, Coordinates(dict(zip("xyz", columns)), clock))
    tolerance = 3e-5 if dtype == torch.float32 else 1e-11
    for actual in [value, equation.laplacian("u")]:
        torch.testing.assert_close(actual, torch.zeros_like(actual), atol=tolerance, rtol=0)
    for axis in "xyz"[:dimensions]:
        actual = equation.diff("u", axis)
        torch.testing.assert_close(actual, torch.zeros_like(actual), atol=tolerance, rtol=0)


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_full_chain_rule_includes_mixed_feature_derivative(dimensions: int) -> None:
    columns, clock = physical_points(dimensions, torch.float64)
    features = NeumannFeatures(BOUNDS[:dimensions], "neumann_polynomial")(columns)
    value, expected_laplacian = torch.zeros_like(clock), torch.zeros_like(clock)
    missing_cross = torch.zeros_like(clock)
    for x, (a, b), q, p in zip(columns, BOUNDS, features[::2], features[1::2]):
        z = (x - a) / (b - a)
        dq, ddq = 6 * z * (1 - z), 6 - 12 * z
        dp, ddp = 32 * z * (1 - z) * (1 - 2 * z), 32 * (1 - 6 * z + 6 * z.square())
        value = value + q * p
        cross = 2 * dq * dp / (b - a) ** 2
        expected_laplacian = expected_laplacian + (ddq * p + q * ddp) / (b - a) ** 2 + cross
        missing_cross = missing_cross + cross
    e = Equation({"u": clock.square() * value}, Coordinates(dict(zip("xyz", columns)), clock))
    torch.testing.assert_close(e.laplacian("u"), clock.square() * expected_laplacian)
    torch.testing.assert_close(e.dt("u"), 2 * clock * value)
    assert float(missing_cross.detach().abs().max()) > 0.01


def affine(c: Coordinates) -> TensorFields:
    return {"u": 1 + sum(c.spatial.values(), torch.zeros_like(c.t))}


def config(dimensions: int, encoding: str) -> dict[str, Any]:
    case = CASES[f"heat_neumann_polynomial{dimensions}d"]
    result = case.config(os.environ.get("PINN_TEST_DEVICE", "cpu"), 2, 19)
    result["network"].update(hidden_layers=[8, 8], spatial_encoding=encoding)
    result["training"]["constraints"] = "hard_initial_robin"
    for axis, (a, b) in zip(result["domain"]["space"].values(), BOUNDS):
        axis.update(interval=[a, b], points=3)
    result["domain"]["time"].update(interval=[0.7, 1.9], points=3)
    result["boundaries"] = [
        dict(kind="neumann", field="u", axis=axis, side=side, value=value)
        for axis in "xyz"[:dimensions]
        for side, value in (("lower", -1), ("upper", 1))
    ]
    return result


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("encoding", ["neumann_cosine", "neumann_polynomial"])
@pytest.mark.parametrize("background", ["general", "auto"])
def test_feature_network_keeps_initial_and_nonzero_neumann_data(
    dimensions: int, encoding: str, background: str, tmp_path: Path
) -> None:
    case = CASES[f"heat_neumann_polynomial{dimensions}d"]
    settings_config = config(dimensions, encoding)
    settings_config["training"]["hard_background"] = background
    settings = TaskSettings.read(settings_config)
    assert settings.training.hard_execution == "eager"
    problem = replace(load_problem(case.source), initial=affine)
    factory = SolverFactory(settings, problem, tmp_path, case.source.parent)
    columns, clock = physical_points(dimensions, torch.float64)
    model = factory.build(True).model.to(device=clock.device, dtype=clock.dtype)
    initial_time = torch.full_like(clock, 0.7)
    torch.testing.assert_close(
        model(columns, initial_time)["u"],
        affine(Coordinates(dict(zip("xyz", columns)), initial_time))["u"],
        rtol=0,
        atol=0,
    )
    physical = dict(model(columns, clock))
    e = Equation(physical, factory.adapter.context(columns, clock))
    corners = 2**dimensions
    for axis in "xyz"[:dimensions]:
        torch.testing.assert_close(
            e.diff("u", axis)[:corners], torch.ones_like(clock[:corners]), atol=2e-12, rtol=0
        )
    loss = (e.dt("u") - 0.1 * e.laplacian("u")).square().mean()
    grads = torch.autograd.grad(loss, tuple(model.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    assert model.hparams["spatial_encoding"] == encoding


def test_cosine_and_polynomial_third_derivative_are_different() -> None:
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float64, requires_grad=True)
    for kind in ("neumann_cosine", "neumann_polynomial"):
        first = NeumannFeatures(((0.0, 1.0),), kind)([x])[0]
        third = first
        for _ in range(3):
            third = torch.autograd.grad(third.sum(), x, create_graph=True)[0]
        expected = torch.zeros_like(x) if kind == "neumann_cosine" else torch.full_like(x, -12)
        torch.testing.assert_close(third, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("invalid", ["soft", "dirichlet", "robin", "grouped", "unknown"])
def test_feature_encoding_rejects_unproved_conditions(invalid: str) -> None:
    value = config(1, "neumann_polynomial")
    if invalid == "soft":
        value["training"]["constraints"] = "soft"
    elif invalid == "dirichlet":
        value["training"]["constraints"] = "hard_initial_dirichlet"
    elif invalid == "robin":
        value["boundaries"][0].update(kind="robin", alpha=1, beta=1)
    elif invalid == "grouped":
        value["training"]["hard_execution"] = "robin_grouped"
    else:
        value["network"]["spatial_encoding"] = "unknown"
    with pytest.raises(ValueError):
        TaskSettings.read(value)


def test_representation_has_separate_signature_but_resolution_does_not(tmp_path: Path) -> None:
    case = CASES["heat_neumann_polynomial1d"]
    problem = replace(load_problem(case.source), initial=affine)
    signatures = []
    for encoding in ("identity", "neumann_cosine", "neumann_polynomial"):
        value = config(1, encoding)
        factory = SolverFactory(TaskSettings.read(value), problem, tmp_path, case.source.parent)
        signature = factory.constraint_signature
        signatures.append(signature)
        value["domain"]["space"]["x"]["points"] = 5
        other = SolverFactory(TaskSettings.read(value), problem, tmp_path, case.source.parent)
        assert other.constraint_signature == signature
    assert len(set(signatures)) == 3


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("encoding", ["neumann_cosine", "neumann_polynomial"])
@pytest.mark.parametrize("lifting", ["recursive", "expanded"])
def test_time_dependent_neumann_continuation_and_prediction(
    dimensions: int, encoding: str, lifting: str, tmp_path: Path
) -> None:
    case = CASES["heat_neumann" if dimensions == 1 else f"heat_neumann{dimensions}d"]
    value = case.config(os.environ.get("PINN_TEST_DEVICE", "cpu"), 2, 19)
    value["network"].update(hidden_layers=[4, 4], spatial_encoding=encoding)
    value["training"].update(constraints="hard_initial_robin", hard_lifting=lifting)
    factory = SolverFactory(
        TaskSettings.read(value), load_problem(case.source), tmp_path, case.source.parent
    )
    device = value["runtime"]["device"]
    unit = torch.tensor(
        list(product((0.0, 1.0), repeat=dimensions)), dtype=torch.float64, device=device
    )
    columns = [unit[:, i : i + 1].clone().requires_grad_() for i in range(dimensions)]
    clock = torch.full_like(columns[0], 0.3, requires_grad=True)
    model = factory.build(True).model.to(device=device, dtype=torch.float64).eval()
    fields = dict(model(columns, clock))
    e = Equation(fields, factory.adapter.context(columns, clock))
    for axis in "xyz"[:dimensions]:
        torch.testing.assert_close(e.diff("u", axis), clock, atol=2e-12, rtol=0)
    # Prediction may disable coordinate gradients; continuation still needs its internal traces.
    predicted = model.predict_step(([x.detach() for x in columns], clock.detach(), {}), 0)
    torch.testing.assert_close(predicted["u"], fields["u"], atol=2e-12, rtol=0)
    assert not predicted["u"].requires_grad
    assert model.hparams["hard_background"] == "general"


@pytest.mark.parametrize("bounds", [(), ((1.0, 1.0),), ((float("-inf"), 1.0),)])
def test_feature_bounds_are_finite_and_increasing(bounds: tuple[tuple[float, float], ...]) -> None:
    with pytest.raises(ValueError):
        NeumannFeatures(bounds, "neumann_polynomial")


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("lifting", ["recursive", "expanded"])
def test_time_and_tangential_data_keep_mixed_corner_traces(
    dimensions: int, lifting: str, tmp_path: Path
) -> None:
    from collections.abc import Callable

    case = CASES[f"heat_neumann_polynomial{dimensions}d"]
    value = case.config(os.environ.get("PINN_TEST_DEVICE", "cpu"), 2, 19)
    value["network"].update(hidden_layers=[4, 4], spatial_encoding="neumann_polynomial")
    value["training"].update(constraints="hard_initial_robin", hard_lifting=lifting)

    def wall_for(axis: str) -> Callable[[Coordinates], torch.Tensor]:
        def wall(c: Coordinates) -> torch.Tensor:
            x = c.spatial[axis]
            result = 2 * c.t * x * (2 * x - 1)
            for name, column in c.spatial.items():
                if name != axis:
                    result = result * column.square()
            return result

        return wall

    names = "xyz"[:dimensions]
    walls = {f"wall_{axis}": wall_for(axis) for axis in names}
    problem = replace(
        load_problem(case.source),
        initial=lambda c: {"u": torch.zeros_like(c.t)},
        boundary_values=walls,
    )
    value["boundaries"] = [
        dict(kind="neumann", field="u", axis=axis, value_function=f"wall_{axis}") for axis in names
    ]
    factory = SolverFactory(TaskSettings.read(value), problem, tmp_path, case.source.parent)
    device = value["runtime"]["device"]
    unit = torch.tensor(
        list(product((0.0, 0.4, 1.0), repeat=dimensions)), dtype=torch.float64, device=device
    )
    columns = [unit[:, i : i + 1].clone().requires_grad_() for i in range(dimensions)]
    clock = torch.full_like(columns[0], 0.3, requires_grad=True)
    c = factory.adapter.context(columns, clock)
    model = factory.build(True).model.to(device=device, dtype=torch.float64).eval()
    physical = dict(model(columns, clock))
    e = Equation(physical, c)
    for i, axis in enumerate(names):
        face = columns[i].eq(0) | columns[i].eq(1)
        normal = columns[i] * 2 - 1
        observed = e.diff("u", axis) * normal
        torch.testing.assert_close(
            observed[face], walls[f"wall_{axis}"](c)[face], atol=1e-11, rtol=0
        )
    predicted = model.predict_step(([x.detach() for x in columns], clock.detach(), {}), 0)
    torch.testing.assert_close(predicted["u"], physical["u"], atol=1e-11, rtol=0)


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("encoding", ["neumann_cosine", "neumann_polynomial"])
def test_exact_last_bias_gradient_of_physical_heat_loss(
    dimensions: int, encoding: str, tmp_path: Path
) -> None:
    case = CASES[f"heat_neumann_polynomial{dimensions}d"]
    value = config(dimensions, encoding)
    problem = replace(load_problem(case.source), initial=affine)
    factory = SolverFactory(TaskSettings.read(value), problem, tmp_path, case.source.parent)
    columns, clock = physical_points(dimensions, torch.float64)
    model = factory.build(True).model.to(device=clock.device, dtype=clock.dtype)
    residual = factory.adapter.residuals(dict(model(columns, clock)), *columns, clock)["heat"]
    # Last output bias contributes b*tau, hence d(residual)/db=1/(t1-t0).
    expected = 2 * residual.detach().mean() / (1.9 - 0.7)
    model.backward(residual.square().mean())
    bias = dict(model.named_parameters())["net.network.model.output.bias"]
    assert bias.grad is not None
    torch.testing.assert_close(bias.grad, expected.reshape_as(bias), atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("execution", ["eager", "neumann_analytic"])
def test_encoded_convergence_evaluates_the_same_representation(
    tmp_path: Path, execution: str
) -> None:
    from benchmarks.convergence import ConvergenceArm, ConvergencePlan, convergence_trial
    from benchmarks.policies import CartesianGrid, TrainingBudget

    case = replace(
        CASES["heat_neumann_polynomial2d"],
        grid=CartesianGrid((3, 3), (3, 3)),
        budget=TrainingBudget(layers=(4, 4), collocation_points=7),
    )
    plan = ConvergencePlan(case, 3, (0, 1, 2), {"relative_l2": 1}, "hard_initial_robin", 7, "cpu")
    arm = ConvergenceArm(execution, "recursive", "parameters", "auto", "neumann_polynomial")
    result = convergence_trial(plan, arm, 19, tmp_path / arm.name)
    assert result["spatial_encoding"] == "neumann_polynomial"
    assert result["resolved_background"] == "static"
    assert result["control_prediction_background"] == "general"
    assert result["prediction_execution"] == execution
    assert result["feature_derivatives"] == (
        "analytic" if execution == "neumann_analytic" else "autograd"
    )
    assert result["control_feature_derivatives"] == "autograd"
    assert result["trainable_parameters"] > 0
    assert len(result["snapshots"][-1]["boundary_operators"]) == 4
