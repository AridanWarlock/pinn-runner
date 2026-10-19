"""Independent conservative-operator controls on exact fields and actual PINN weights."""

from pathlib import Path

import pytest
import torch

from pinnstorch.api import Coordinates, Equation
from scripts.audit_variable_conductivity import audit_pair, make_factory, points


@pytest.mark.parametrize("dimensions", [1, 2])
@pytest.mark.parametrize("beta", [0, 1])
def test_every_network_parameter_matches_expanded_operator(
    tmp_path: Path, dimensions: int, beta: int
) -> None:
    rows = audit_pair(tmp_path / "pair", dimensions, torch.float64, beta, 19)
    assert len(rows) == 2
    assert all(row["state_unchanged"] for row in rows)
    assert all(row["parameters"] > 2000 for row in rows)
    assert all(row["directional_absolute_error"] is not None for row in rows)


@pytest.mark.parametrize("dimensions", [1, 2])
@pytest.mark.parametrize("representation", ["formula", "table"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_field_detects_missing_conductivity_derivative(
    tmp_path: Path, dimensions: int, representation: str, dtype: torch.dtype
) -> None:
    factory = make_factory(tmp_path / "case", dimensions, 1, representation, 19)
    factory.adapter.coefficients.to(dtype=dtype)
    values = points(dimensions, dtype)
    columns = [values[:, i : i + 1].clone().requires_grad_() for i in range(dimensions + 1)]
    context = factory.adapter.context(columns[:-1], columns[-1])
    exact = factory.problem.initial(context)["u"] * (-context.t).exp()
    residual = factory.adapter.residuals({"u": exact}, *columns)["heat"]
    tolerance = 5e-6 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=tolerance)
    equation = Equation({"u": exact}, context)
    correct_divergence = equation.divergence(
        {
            axis: context.coefficients["conductivity"] * derivative
            for axis, derivative in equation.gradient("u").items()
        }
    )
    wrong_divergence = context.coefficients["conductivity"] * equation.laplacian("u")
    omitted = correct_divergence - wrong_divergence
    # Closed expression for grad(kappa).grad(u), independent of Equation.gradient.
    expected = torch.zeros_like(context.t)
    for i, x in enumerate(columns[:-1]):
        term = 6 * x * (1 - x)
        for j, y in enumerate(columns[:-1]):
            if j != i:
                term = term * (3 * y.square() - 2 * y.pow(3))
        expected = expected + term
    expected = expected * (-context.t).exp() / (8 * dimensions)
    torch.testing.assert_close(omitted, expected, rtol=2e-4, atol=tolerance)
    assert float(omitted.abs().max()) > 0.05
    # The initial temperature has the prescribed physical axes and shape.
    c = Coordinates(dict(zip("xy"[:dimensions], columns[:-1])), columns[-1])
    assert factory.problem.initial(c)["u"].shape == columns[-1].shape
