"""Analytic and backpropagation checks for the public equation language."""

from unittest.mock import patch

import pytest
import torch

from pinnstorch.api import Coordinates, Equation


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_heat_equation_and_cached_derivatives(dimension: int) -> None:
    spatial = {
        axis: torch.linspace(0.1, 0.9, 7, dtype=torch.float64).reshape(-1, 1).requires_grad_()
        for axis in ("x", "y", "z")[:dimension]
    }
    time = torch.full_like(spatial["x"], 0.2).requires_grad_()
    coefficient = torch.linspace(0.1, 0.5, 7, dtype=torch.float64).reshape(-1, 1)
    shape = torch.ones_like(time)
    for coordinate in spatial.values():
        shape = shape * torch.sin(torch.pi * coordinate)
    temperature = shape * torch.exp(-coefficient * dimension * torch.pi**2 * time)
    equation = Equation({"T": temperature}, Coordinates(spatial, time, {"k": coefficient}))
    with patch("torch.autograd.grad", wraps=torch.autograd.grad) as differentiate:
        residual = equation.dt("T") - coefficient * equation.laplacian("T")
        torch.testing.assert_close(residual, torch.zeros_like(time), atol=1e-14, rtol=0)
        assert differentiate.call_count == dimension + 1
        assert equation.dt("T") is equation.dt("T")
        equation.gradient("T")
        equation.laplacian("T")
        assert differentiate.call_count == dimension + 1


def test_variable_conductivity_and_mixed_derivatives() -> None:
    x = torch.linspace(0.1, 0.9, 5, dtype=torch.float64).reshape(-1, 1).requires_grad_()
    y = (x.detach() + 0.2).requires_grad_()
    time = torch.ones_like(x).requires_grad_()
    equation = Equation({"T": x.square() + y.square()}, Coordinates({"x": x, "y": y}, time))
    conductivity = 1 + x
    flux = {axis: conductivity * derivative for axis, derivative in equation.gradient("T").items()}
    torch.testing.assert_close(equation.divergence(flux), 4 + 6 * x)
    torch.testing.assert_close(equation.diff(equation.diff("T", "x"), "y"), torch.zeros_like(x))


def test_backpropagation_through_second_derivatives() -> None:
    x = torch.linspace(0.1, 0.9, 5).reshape(-1, 1).requires_grad_()
    time = torch.ones_like(x).requires_grad_()
    scale = torch.tensor(2.0, requires_grad=True)
    equation = Equation({"T": scale * x.square()}, Coordinates({"x": x}, time))
    equation.laplacian("T").square().mean().backward()
    torch.testing.assert_close(scale.grad, torch.tensor(16.0))


def test_constant_fields_and_new_context_do_not_share_cache() -> None:
    x = torch.ones(3, 1, requires_grad=True)
    time = torch.ones_like(x).requires_grad_()
    coordinates = Coordinates({"x": x}, time)
    first = Equation({"T": x.square()}, coordinates)
    second = Equation({"T": 3 * x.square()}, coordinates)
    torch.testing.assert_close(first.laplacian("T"), torch.full_like(x, 2))
    torch.testing.assert_close(second.laplacian("T"), torch.full_like(x, 6))
    constant = Equation({"T": torch.ones_like(x)}, coordinates)
    torch.testing.assert_close(constant.laplacian("T"), torch.zeros_like(x))


def test_invalid_differentiation_requests() -> None:
    x = torch.ones(3, 1, requires_grad=True)
    equation = Equation({"T": x.square()}, Coordinates({"x": x}, x))
    with pytest.raises(ValueError, match="Unknown"):
        equation.diff("T", "not_an_axis")
    with pytest.raises(ValueError, match="order"):
        equation.diff("T", "x", 0)
    with pytest.raises(ValueError, match="each physical"):
        equation.divergence({})


def test_equations_share_derivative_cache_through_adapter() -> None:
    from pinnstorch.api import Problem
    from pinnstorch.runner.problem import ProblemAdapter
    from pinnstorch.runner.settings import TaskSettings

    settings = TaskSettings.read(
        {
            "version": 1,
            "domain": {
                "space": {"x": {"interval": [0, 1], "points": 3}},
                "time": {"interval": [0, 1], "points": 2},
            },
        }
    )
    problem = Problem(
        fields=("T",),
        equations={"first": lambda e: e.dt("T"), "second": lambda e: e.dt("T") + e.laplacian("T")},
        initial=lambda c: {"T": torch.zeros_like(c.t)},
    )
    adapter = ProblemAdapter(problem, settings.domain)
    x = torch.ones(3, 1, requires_grad=True)
    time = torch.ones(3, 1, requires_grad=True)
    with patch("torch.autograd.grad", wraps=torch.autograd.grad) as differentiate:
        outputs = adapter.residuals({"T": x.square() + time}, x, time)
        assert differentiate.call_count == 2
    torch.testing.assert_close(outputs["first"], torch.ones_like(time))
    torch.testing.assert_close(outputs["second"], torch.full_like(time, 3))
