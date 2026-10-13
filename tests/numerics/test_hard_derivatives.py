"""Product-rule heat derivatives versus full AD, including trainable IC."""

import os

import pytest
import torch

from pinnstorch.api import Coordinates, Equation, TensorFields
from pinnstorch.training.constraints import HardBoxInitial
from pinnstorch.training.hard_derivatives import HardBoxDerivatives


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_product_rule_preserves_derivatives_residual_and_weight_gradients(
    dimension: int, dtype: torch.dtype
) -> None:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    bounds = [(float(i - 2), float(2 * i + 1)) for i in range(dimension)]
    axes = [
        torch.tensor([left, left + 1e-5, (left + right) / 2, right], dtype=dtype, device=device)
        for left, right in bounds
    ]
    grid = torch.cartesian_prod(
        *axes, torch.tensor([2.0, 2.1, 4.3, 5.0], dtype=dtype, device=device)
    )
    columns = [grid[:, i : i + 1].detach().requires_grad_() for i in range(dimension + 1)]
    weight = torch.tensor(0.7, dtype=dtype, device=device, requires_grad=True)
    coordinates = Coordinates(dict(zip(("x", "y", "z"), columns[:-1])), columns[-1])

    def initial(spatial: list[torch.Tensor], time: torch.Tensor) -> TensorFields:
        value = torch.ones_like(time) * weight
        for column, (left, right) in zip(spatial, bounds):
            value = value * (column - left) * (right - column)
        return {"u": value}

    network = {"u": torch.sin(sum(columns) * weight) + weight**2}
    transform = HardBoxInitial(bounds, (2, 5), initial)
    expected = Equation(transform(network, *columns), coordinates)
    candidate = HardBoxDerivatives(transform, network, coordinates)
    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(
        candidate.field("u"), expected.field("u"), rtol=tolerance, atol=tolerance
    )
    for axis in (*coordinates.spatial, "t"):
        for order in (1, 2):
            torch.testing.assert_close(
                candidate.diff("u", axis, order),
                expected.diff("u", axis, order),
                rtol=tolerance,
                atol=tolerance,
            )
    residual = candidate.heat_residual("u", 0.1)
    reference = expected.dt("u") - 0.1 * expected.laplacian("u")
    torch.testing.assert_close(residual, reference, rtol=tolerance, atol=tolerance)
    actual_grad = torch.autograd.grad(residual.square().mean(), weight, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(reference.square().mean(), weight)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=tolerance, atol=tolerance)
