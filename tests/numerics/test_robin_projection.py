"""Boundary operators vanish while temperature and training derivatives stay free."""

import os

import pytest
import torch

from pinnstorch.api import Coordinates, Equation
from pinnstorch.contracts import TensorOutputs
from pinnstorch.training.robin_projection import IntervalRobinProjection, RobinOperator


@pytest.mark.parametrize("alphas", [(0.0, 0.0), (2.0, 3.0), (-1.0, 0.0)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_projection_preserves_free_temperature_and_parameter_gradients(
    alphas: tuple[float, float], dtype: torch.dtype
) -> None:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    projector = IntervalRobinProjection(
        (-2, 3), RobinOperator(alphas[0], 0.7), RobinOperator(alphas[1], -1.2)
    )
    weight = torch.tensor(0.6, dtype=dtype, device=device, requires_grad=True)
    x = torch.tensor(
        [[-2.0], [-2 + 1e-5], [0.2], [3.0]], dtype=dtype, device=device, requires_grad=True
    )
    time = torch.full_like(x, 1.3, requires_grad=True)

    def field(coordinate: torch.Tensor, clock: torch.Tensor) -> TensorOutputs:
        return {"u": coordinate.square().mul(weight).add(clock.sin().mul(weight)).add(2)}

    projected = projector(field, x, time)
    equation = Equation(projected, Coordinates({"x": x}, time))
    original = field(x, time)["u"]
    tolerance = 2e-12 if dtype == torch.float64 else 2e-5
    # Hermite correction changes boundary derivatives, not boundary values.
    torch.testing.assert_close(projected["u"][[0, -1]], original[[0, -1]], rtol=0, atol=0)
    derivative = equation.diff("u", "x")
    torch.testing.assert_close(
        projected["u"][0] * alphas[0] - 0.7 * derivative[0],
        torch.zeros_like(derivative[0]),
        rtol=0,
        atol=tolerance,
    )
    torch.testing.assert_close(
        projected["u"][-1] * alphas[1] - 1.2 * derivative[-1],
        torch.zeros_like(derivative[-1]),
        rtol=0,
        atol=tolerance,
    )
    # Independent polynomial form of the correction, including theta dependence.
    s = (x + 2) / 5
    phi_lower = -5 / 0.7 * s * (1 - s) ** 2
    phi_upper = 5 / -1.2 * s**2 * (s - 1)
    lower = alphas[0] * (4 * weight + time.sin() * weight + 2) + 2.8 * weight
    upper = alphas[1] * (9 * weight + time.sin() * weight + 2) - 7.2 * weight
    reference = Equation(
        {"u": original - phi_lower * lower - phi_upper * upper}, Coordinates({"x": x}, time)
    )
    for axis in ("x", "t"):
        for order in (1, 2):
            actual, expected = equation.diff("u", axis, order), reference.diff("u", axis, order)
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(
                torch.autograd.grad(actual.square().mean(), weight, retain_graph=True)[0],
                torch.autograd.grad(expected.square().mean(), weight, retain_graph=True)[0],
                rtol=tolerance,
                atol=tolerance,
            )
    residual = equation.dt("u") - 0.1 * equation.diff("u", "x", 2)
    expected_residual = reference.dt("u") - 0.1 * reference.diff("u", "x", 2)
    torch.testing.assert_close(residual, expected_residual, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        torch.autograd.grad(residual.square().mean(), weight, retain_graph=True)[0],
        torch.autograd.grad(expected_residual.square().mean(), weight, retain_graph=True)[0],
        rtol=tolerance,
        atol=tolerance,
    )

    def already_projected(coordinate: torch.Tensor, clock: torch.Tensor) -> TensorOutputs:
        return projector(field, coordinate, clock)

    torch.testing.assert_close(
        projector(already_projected, x, time)["u"], projected["u"], rtol=tolerance, atol=tolerance
    )
    with torch.no_grad():
        prediction = projector(field, x.detach(), time.detach())["u"]
        assert not prediction.requires_grad
        torch.testing.assert_close(prediction, projected["u"], rtol=0, atol=0)


def test_projection_rejects_dirichlet_and_invalid_interval() -> None:
    with pytest.raises(ValueError, match="nonzero beta"):
        RobinOperator(1, 0)
    with pytest.raises(ValueError, match="positive interval"):
        IntervalRobinProjection((2, 2), RobinOperator(0, 1), RobinOperator(0, 1))
