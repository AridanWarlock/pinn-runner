"""Product-rule heat derivatives versus full AD, including trainable IC."""

import os

import pytest
import torch

from pinnstorch.api import Coordinates, Equation, TensorFields
from pinnstorch.training.constraints import HardBoxInitial
from pinnstorch.training.hard_derivatives import HardBoxDerivatives
from pinnstorch.training.hard_execution import ProductRuleHard


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


@pytest.mark.parametrize("order", [1, 2, 3])
def test_equation_dispatch_and_higher_order_fallback(order: int) -> None:
    x = torch.linspace(-2, 1, 7, dtype=torch.float64).reshape(-1, 1).requires_grad_()
    t = torch.full_like(x, 2.4, requires_grad=True)
    coordinates = Coordinates({"radius": x}, t)
    weight = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    fields = {"u": torch.sin(x * weight + t)}

    def initial(spatial: list[torch.Tensor], time: torch.Tensor) -> TensorFields:
        return {"u": spatial[0].square().mul(weight)}

    eager = HardBoxInitial([(-2, 1)], (2, 5), initial)
    prepared = ProductRuleHard(
        HardBoxInitial([(-2, 1)], (2, 5), initial, spatial_names=("radius",))
    )
    expected = Equation(eager(fields, x, t), coordinates)
    actual = Equation(prepared(fields, x, t), coordinates)
    torch.testing.assert_close(actual.field("u"), expected.field("u"), rtol=0, atol=0)
    derivative = actual.diff("u", "radius", order)
    reference = expected.diff("u", "radius", order)
    torch.testing.assert_close(derivative, reference, rtol=1e-12, atol=1e-12)
    assert actual.diff("u", "radius", order) is derivative
    torch.testing.assert_close(
        torch.autograd.grad(derivative.square().mean(), weight, retain_graph=True)[0],
        torch.autograd.grad(reference.square().mean(), weight)[0],
        rtol=1e-12,
        atol=1e-12,
    )
    with torch.no_grad():
        torch.testing.assert_close(
            prepared(fields, x, t)["u"], eager(fields, x, t)["u"], rtol=0, atol=0
        )


def test_constant_fields_and_fresh_graph_after_parameter_update() -> None:
    weight = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)

    def initial(spatial: list[torch.Tensor], time: torch.Tensor) -> TensorFields:
        return {"u": torch.zeros_like(time)}

    eager = HardBoxInitial([(0, 1)], (2, 5), initial)
    prepared = ProductRuleHard(eager)
    for iteration in range(3):
        x = torch.tensor(
            [[0.0], [0.2 + 0.1 * iteration], [1.0]], dtype=torch.float64, requires_grad=True
        )
        t = torch.full_like(x, 2.4 + iteration * 0.1, requires_grad=True)
        coordinates = Coordinates({"x": x}, t)
        fields = {"u": torch.ones_like(x) * weight}
        actual = Equation(prepared(fields, x, t), coordinates)
        reference = Equation(eager(fields, x, t), coordinates)
        residual = actual.dt("u") - 0.1 * actual.laplacian("u")
        expected = reference.dt("u") - 0.1 * reference.laplacian("u")
        torch.testing.assert_close(residual, expected, rtol=1e-12, atol=1e-12)
        gradient = torch.autograd.grad(residual.square().mean(), weight, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected.square().mean(), weight)[0]
        torch.testing.assert_close(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
        with torch.no_grad():
            weight.add_(-0.1 * gradient)
