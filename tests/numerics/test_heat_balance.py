"""Integral diagnostics agree with the manufactured PDE and detect added drift."""

import math

import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.heat_balance import heat_balance
from pinnstorch.api import Coordinates
from pinnstorch.runner.problem import load_problem


@pytest.mark.parametrize("name", ["heat_neumann", "heat_robin"])
def test_analytical_heat_balance_and_added_drift(name: str) -> None:
    problem = load_problem(CASES[name].source)
    mu = math.pi / 2 if name == "heat_robin" else math.pi

    def field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shifted = x - 0.5 if name == "heat_robin" else x
        return 1 + x * t + torch.exp(-0.1 * mu**2 * t) * torch.cos(mu * shifted)

    def target(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return problem.boundary_values["wall"](Coordinates({"x": x}, t))

    def source(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x

    def wrong(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return field(x, t) + t * 0.1

    kwargs = dict(
        bounds=(0.0, 1.0),
        time_interval=(0.0, 0.5),
        device="cpu",
        alpha=mu if name == "heat_robin" else 0,
        beta=1,
        target=target,
        source=source,
    )
    metrics = heat_balance(field, **kwargs)
    assert metrics["boundary_operator_rmse"] < 1e-5
    assert metrics["heat_balance_rate_rmse"] < 1e-5
    assert metrics["heat_balance_accumulated_max_error"] < 1e-5
    incorrect = heat_balance(wrong, **kwargs)
    assert incorrect["heat_balance_rate_rmse"] > 0.09
    assert incorrect["heat_balance_accumulated_max_error"] > 0.045
