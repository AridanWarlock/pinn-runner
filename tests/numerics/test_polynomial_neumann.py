"""A smooth Neumann counterexample and compatibility of the existing hard path."""

from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.cases import CASES
from pinnstorch.api import Coordinates, Equation
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_polynomial_has_nonzero_third_normal_derivative(dimensions: int) -> None:
    case = CASES[f"heat_neumann_polynomial{dimensions}d"]
    problem = load_problem(case.source)
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float64, requires_grad=True)
    spatial = {axis: torch.full_like(x, 0.5) for axis in "xyz"[:dimensions]}
    spatial["x"] = x
    c = Coordinates(spatial, torch.full_like(x, 0.3))
    # Initial profile is independently evaluated; time factor has no x dependence.
    physical = problem.initial(c)["u"] * torch.exp(-c.t)
    derivative = physical
    derivatives = []
    for _ in range(3):
        derivative = torch.autograd.grad(derivative.sum(), x, create_graph=True)[0]
        derivatives.append(derivative)
    torch.testing.assert_close(derivatives[0], torch.zeros_like(x), rtol=0, atol=0)
    expected = torch.full_like(x, -12 * 0.5 ** (dimensions - 1)) * torch.exp(-c.t)
    torch.testing.assert_close(derivatives[2], expected, rtol=0, atol=2e-14)
    # Integral of each spatial polynomial is 1/2; the zero-flux balance is dM/dt=-M.
    nodes, weights = np.polynomial.legendre.leggauss(5)
    q = 3 * ((nodes + 1) / 2) ** 2 - 2 * ((nodes + 1) / 2) ** 3
    assert np.dot(weights / 2, q) ** dimensions == pytest.approx(0.5**dimensions, abs=1e-14)


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("execution", ["eager", "robin_grouped"])
def test_polynomial_case_supports_current_hard_constraints(
    dimensions: int, execution: str, tmp_path: Path
) -> None:
    case = CASES[f"heat_neumann_polynomial{dimensions}d"]
    config = case.config("cpu", 2, 19)
    config["network"]["hidden_layers"] = [4, 4]
    config["training"].update(constraints="hard_initial_robin", hard_execution=execution)
    problem = load_problem(case.source)
    factory = SolverFactory(TaskSettings.read(config), problem, tmp_path, case.source.parent)
    model = factory.build(True).model.double()
    for axis in "xyz"[:dimensions]:
        columns = {
            name: torch.full((2, 1), 0.4, dtype=torch.float64, requires_grad=True)
            for name in "xyz"[:dimensions]
        }
        columns[axis] = torch.tensor([[0.0], [1.0]], dtype=torch.float64, requires_grad=True)
        clock = torch.full((2, 1), 0.3, dtype=torch.float64, requires_grad=True)
        c = Coordinates(columns, clock)
        value = model(list(columns.values()), clock)["u"]
        normal_derivative = Equation({"u": value}, c).diff("u", axis)
        torch.testing.assert_close(normal_derivative, torch.zeros_like(clock), rtol=0, atol=1e-12)
        initial_time = torch.zeros_like(clock)
        torch.testing.assert_close(
            model(list(columns.values()), initial_time)["u"],
            problem.initial(Coordinates(columns, initial_time))["u"],
            rtol=0,
            atol=0,
        )
        residual = problem.equations["heat"](Equation({"u": value}, c))
        model.zero_grad(set_to_none=True)
        residual.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
