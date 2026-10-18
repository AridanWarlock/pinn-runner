"""Independent autograd control for analytic feature pullbacks and PDE weight gradients."""

import os
from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest
import torch

from benchmarks.cases import CASES
from pinnstorch.api import Coordinates, TensorFields
from pinnstorch.models.net.neumann_polynomial import analytic_polynomial_features
from pinnstorch.models.net.spatial_features import FeatureFCN, NeumannFeatures
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings


def device() -> str:
    return os.environ.get("PINN_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("second_order", [False, True])
def test_analytic_pullback_passes_independent_finite_difference_checks(second_order: bool) -> None:
    x = torch.tensor([[0.0], [0.17], [0.49], [0.93], [1.0]], device=device(), dtype=torch.float64)
    x.requires_grad_()
    check = torch.autograd.gradgradcheck if second_order else torch.autograd.gradcheck
    assert check(analytic_polynomial_features, (x,))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_analytic_pullback_matches_explicit_physical_derivatives(dtype: torch.dtype) -> None:
    x = torch.tensor([[0.0], [0.13], [0.5], [0.91], [1.0]], dtype=dtype, device=device())
    physical = x.mul(2).sub(0.7).detach().requires_grad_()
    normalized = physical.add(0.7).div(2)
    q, p = NeumannFeatures(((-0.7, 1.3),), "neumann_polynomial", "analytic")([physical])
    r = normalized * (1 - normalized)
    expected = (
        (6 * r / 2, (6 - 12 * normalized) / 4, torch.full_like(x, -12 / 8)),
        (
            32 * r * (1 - 2 * normalized) / 2,
            32 * (1 - 6 * normalized + 6 * normalized.square()) / 4,
            (-192 + 384 * normalized) / 8,
        ),
    )
    for value, derivatives in zip((q, p), expected):
        for derivative in derivatives:
            value = torch.autograd.grad(
                value, physical, torch.ones_like(value), create_graph=True
            )[0]
            torch.testing.assert_close(
                value,
                derivative,
                rtol=3e-5 if dtype == torch.float32 else 1e-12,
                atol=3e-5 if dtype == torch.float32 else 1e-12,
            )


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frozen_physical_field_pde_and_parameter_gradient_match_autograd(
    tmp_path: Path, dimensions: int, dtype: torch.dtype
) -> None:
    case = CASES[f"heat_neumann_homogeneous{dimensions}d"]
    config = case.config(device(), 1, 19)
    config["network"].update(hidden_layers=[6, 6], spatial_encoding="neumann_polynomial")
    config["training"].update(constraints="hard_initial_robin", hard_execution="neumann_analytic")
    factory = SolverFactory(
        TaskSettings.read(config), load_problem(case.source), tmp_path, case.source.parent
    )
    model = factory.build(predicting=True).model.to(device=device(), dtype=dtype)
    assert isinstance(model.net, FeatureFCN)
    assert model.net.features.derivative_rule == "analytic"
    results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for rule in ("autograd", "analytic"):
        model.net.features = replace(model.net.features, derivative_rule=rule)
        model.zero_grad(set_to_none=True)
        values = torch.tensor(
            [0.0, 0.13, 0.31, 0.5, 0.79, 0.93, 1.0], device=device(), dtype=dtype
        )
        columns = [
            values.roll(i).reshape(-1, 1).detach().requires_grad_() for i in range(dimensions)
        ]
        clock = (
            torch.linspace(0.01, 0.49, len(values), device=device(), dtype=dtype)
            .reshape(-1, 1)
            .requires_grad_()
        )
        outputs = dict(model(columns, clock))
        residual = factory.adapter.residuals(dict(outputs), *columns, clock)["heat"]
        first = torch.autograd.grad(
            outputs["u"], (*columns, clock), torch.ones_like(clock), retain_graph=True
        )
        torch.autograd.backward(residual.square().mean(), inputs=tuple(model.parameters()))
        gradient = torch.cat(
            [p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]
        )
        results.append(
            (
                outputs["u"].detach(),
                torch.cat(first, dim=1).detach(),
                residual.detach(),
                gradient.detach().clone(),
            )
        )
    assert torch.equal(results[0][0], results[1][0])
    for control, actual in zip(results[0][1:], results[1][1:]):
        torch.testing.assert_close(
            actual,
            control,
            rtol=2e-4 if dtype == torch.float32 else 1e-10,
            atol=5e-6 if dtype == torch.float32 else 1e-11,
        )


def test_analytic_features_remain_no_grad_during_prediction() -> None:
    x = torch.tensor([[0.2], [0.8]], device=device(), requires_grad=True)
    with torch.no_grad():
        q, p = NeumannFeatures(((0.0, 1.0),), "neumann_polynomial", "analytic")([x])
    assert not q.requires_grad and not p.requires_grad


def test_analytic_rule_rejects_nonpolynomial_encoding() -> None:
    with pytest.raises(ValueError, match="polynomial"):
        NeumannFeatures(((0.0, 1.0),), "neumann_cosine", "analytic")


@pytest.mark.parametrize("encoding", ["identity", "neumann_cosine"])
def test_analytic_execution_requires_polynomial_representation(encoding: str) -> None:
    config = CASES["heat_neumann_homogeneous1d"].config("cpu", 1, 19)
    config["network"]["spatial_encoding"] = encoding
    config["training"].update(constraints="hard_initial_robin", hard_execution="neumann_analytic")
    with pytest.raises(ValueError, match="neumann_polynomial"):
        TaskSettings.read(config)


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shifted_geometry_full_field_identity(
    tmp_path: Path, dimensions: int, dtype: torch.dtype
) -> None:
    """Cover all face intersections, initial time and near-wall physical Hessians."""
    bounds = ((-2.0, 0.5), (0.25, 1.75), (1.0, 3.5))[:dimensions]
    t0, t1 = 0.75, 2.25
    case = CASES[f"heat_neumann_homogeneous{dimensions}d"]
    config = case.config(device(), 1, 19)
    config["network"].update(hidden_layers=[6, 6], spatial_encoding="neumann_polynomial")
    config["training"].update(constraints="hard_initial_robin", hard_execution="neumann_analytic")
    for axis, (a, b) in zip(config["domain"]["space"].values(), bounds):
        axis.update(interval=[a, b], points=3)
    config["domain"]["time"].update(interval=[t0, t1], points=3)

    def initial(c: Coordinates) -> TensorFields:
        value = torch.ones_like(c.t)
        for x, (a, b) in zip(c.spatial.values(), bounds):
            z = (x - a) / (b - a)
            value = value + z.square() * (1 - z).square()
        return {"u": value}

    problem = replace(load_problem(case.source), initial=initial)
    factory = SolverFactory(TaskSettings.read(config), problem, tmp_path, case.source.parent)
    torch.manual_seed(19)
    model = factory.build(predicting=True).model.to(device=device(), dtype=dtype)
    assert isinstance(model.net, FeatureFCN)
    # Tensor product explicitly includes every face, edge and corner at all times.
    rows = list(product(*([(0.0, 0.37, 1.0)] * dimensions), (0.0, 0.43, 1.0)))
    for i in range(dimensions):
        for z in (1e-4, 1 - 1e-4):
            row = [0.37] * dimensions + [0.43]
            row[i] = z
            rows.append(tuple(row))
    unit = torch.tensor(rows, device=device(), dtype=dtype)
    tolerance = (
        dict(rtol=2e-4, atol=5e-6) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-11)
    )
    results = []
    for rule in ("autograd", "analytic"):
        model.net.features = replace(model.net.features, derivative_rule=rule)
        columns = [
            (unit[:, i : i + 1] * (b - a) + a).detach().requires_grad_()
            for i, (a, b) in enumerate(bounds)
        ]
        clock = (unit[:, -1:] * (t1 - t0) + t0).detach().requires_grad_()
        inputs = (*columns, clock)
        field = model(columns, clock)["u"]
        first = torch.autograd.grad(field.sum(), inputs, create_graph=True)
        second = torch.cat(
            [
                torch.cat(torch.autograd.grad(g.sum(), inputs, create_graph=True), dim=1)
                for g in first
            ],
            dim=1,
        )
        residual = factory.adapter.residuals({"u": field}, *columns, clock)["heat"]
        grads = torch.autograd.grad(residual.square().mean(), tuple(model.parameters()))
        # Independent oracle: output bias adds tau*b, hence dR/db = 1/(t1-t0).
        bias_index = [name for name, _ in model.named_parameters()].index(
            "net.network.model.output.bias"
        )
        torch.testing.assert_close(
            grads[bias_index],
            (2 * residual.detach().mean() / (t1 - t0)).reshape_as(grads[bias_index]),
            **tolerance,
        )
        results.append(
            tuple(
                v.detach()
                for v in (
                    field,
                    torch.cat(first, dim=1),
                    second,
                    residual,
                    torch.cat([g.reshape(-1) for g in grads]),
                )
            )
        )
        start = unit[:, -1:].eq(0)
        expected = initial(factory.adapter.context(columns, clock))["u"]
        torch.testing.assert_close(field[start], expected[start], rtol=0, atol=0)
        for i in range(dimensions):
            for side, sign in ((0, -1), (1, 1)):
                face = unit[:, i : i + 1].eq(side)
                normal_derivative = sign * first[i][face]
                torch.testing.assert_close(
                    normal_derivative, torch.zeros_like(normal_derivative), **tolerance
                )
        assert all(torch.isfinite(v).all() for v in results[-1])
        assert results[-1][-1].abs().max() > 0
    assert torch.equal(results[0][0], results[1][0])
    for control, actual in zip(results[0][1:], results[1][1:]):
        torch.testing.assert_close(actual, control, **tolerance)
