"""Batched normal/mixed jets preserve the eager physical field and training graph."""

import os

import pytest
import torch

from pinnstorch.api import Coordinates, Equation
from pinnstorch.contracts import TensorOutputs
from pinnstorch.training.robin_batch import BatchedBoxRobinProjection
from pinnstorch.training.robin_box import BoxRobinProjection
from pinnstorch.training.robin_grouped import GroupedBoxRobinProjection
from pinnstorch.training.robin_projection import IntervalRobinProjection, RobinOperator


@pytest.mark.parametrize("neumann", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_projection_matches_eager_nonlinear_field_and_gradients(
    dimensions: int, dtype: torch.dtype, grouped: bool, neumann: bool
) -> None:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    torch.manual_seed(19)
    axes = tuple(
        IntervalRobinProjection(
            (-0.5 + 0.2 * i, 0.7 + 0.5 * i),
            RobinOperator(0 if neumann else i, 0.8),
            RobinOperator(0 if neumann else 2, -1.1),
        )
        for i in range(dimensions)
    )
    eager = BoxRobinProjection(axes)
    batched = GroupedBoxRobinProjection(axes) if grouped else BatchedBoxRobinProjection(axes)
    network = torch.nn.Sequential(
        torch.nn.Linear(dimensions + 1, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
    ).to(device=device, dtype=dtype)
    spatial = [
        torch.linspace(*axis.bounds, 7, device=device, dtype=dtype).reshape(-1, 1).requires_grad_()
        for axis in axes
    ]
    time = torch.linspace(0, 0.5, 7, device=device, dtype=dtype).reshape(-1, 1).requires_grad_()
    calls = 0

    def field(columns: list[torch.Tensor], clock: torch.Tensor) -> TensorOutputs:
        nonlocal calls
        calls += 1
        return {"u": network(torch.cat([*columns, clock], dim=1))}

    original = field(spatial, time)
    reference = eager.apply(original, field, spatial, time)
    calls = 0
    actual = batched.apply(original, field, spatial, time)
    assert calls == (2**dimensions - 1 if grouped else 1)
    tolerance = 3e-5 if dtype == torch.float32 else 5e-12
    coordinates = Coordinates(dict(zip("xyz", spatial)), time)
    expected, equation = Equation(reference, coordinates), Equation(actual, coordinates)
    torch.testing.assert_close(actual["u"], reference["u"], atol=tolerance, rtol=tolerance)
    for axis in (*"xyz"[:dimensions], "t"):
        for order in (1, 2):
            torch.testing.assert_close(
                equation.diff("u", axis, order),
                expected.diff("u", axis, order),
                atol=tolerance,
                rtol=tolerance,
            )
    if dimensions > 1:
        torch.testing.assert_close(
            equation.diff(equation.diff("u", "x"), "y"),
            expected.diff(expected.diff("u", "x"), "y"),
            atol=tolerance,
            rtol=tolerance,
        )
    residual = equation.dt("u") - 0.1 * equation.laplacian("u")
    reference_residual = expected.dt("u") - 0.1 * expected.laplacian("u")
    gradients = torch.autograd.grad(
        residual.square().mean(), tuple(network.parameters()), retain_graph=True, allow_unused=True
    )
    reference_gradients = torch.autograd.grad(
        reference_residual.square().mean(),
        tuple(network.parameters()),
        retain_graph=True,
        allow_unused=True,
    )
    for parameter, gradient, reference_gradient in zip(
        network.parameters(), gradients, reference_gradients
    ):
        # A removed zero path may be absent instead of returning a zero Tensor.
        actual_gradient = torch.zeros_like(parameter) if gradient is None else gradient
        expected_gradient = (
            torch.zeros_like(parameter) if reference_gradient is None else reference_gradient
        )
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance
        )
    with torch.no_grad():
        prediction = batched.apply(
            field([x.detach() for x in spatial], time.detach()),
            field,
            [x.detach() for x in spatial],
            time.detach(),
        )["u"]
        assert not prediction.requires_grad
        torch.testing.assert_close(prediction, actual["u"], atol=tolerance, rtol=tolerance)
