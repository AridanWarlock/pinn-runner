"""Independent autograd control for analytic feature pullbacks and PDE weight gradients."""

import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from benchmarks.cases import CASES
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
    config["training"].update(constraints="hard_initial_robin", hard_execution="eager")
    factory = SolverFactory(
        TaskSettings.read(config), load_problem(case.source), tmp_path, case.source.parent
    )
    model = factory.build(predicting=True).model.to(device=device(), dtype=dtype)
    assert isinstance(model.net, FeatureFCN)
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
