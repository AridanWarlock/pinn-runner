"""Expanded data lifting preserves field jets, normal traces and training gradients."""

import itertools
import os
from functools import partial

import pytest
import torch

from pinnstorch.api import Coordinates, Equation
from pinnstorch.contracts import TensorOutputs
from pinnstorch.training.robin_box import BoxRobinInitial, BoxRobinProjection
from pinnstorch.training.robin_lifting import ExpandedRobinLifting
from pinnstorch.training.robin_projection import IntervalRobinProjection, RobinOperator


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("neumann", [False, True])
@pytest.mark.parametrize("paired_time", [False, True])
def test_expanded_lifting_matches_recursive_physical_derivatives(
    dimensions: int, dtype: torch.dtype, neumann: bool, paired_time: bool
) -> None:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    axes = tuple(
        IntervalRobinProjection(
            (-0.4 + i * 0.1, 1.1 + i * 0.3),
            RobinOperator(0 if neumann else i + 0.2, 0.7),
            RobinOperator(0 if neumann else 1.1 - i, -0.9),
        )
        for i in range(dimensions)
    )
    locations = list(
        itertools.product(*[(a.bounds[0], sum(a.bounds) / 2, a.bounds[1]) for a in axes])
    )
    spatial = [
        torch.tensor([[row[i]] for row in locations], dtype=dtype, device=device).requires_grad_()
        for i in range(dimensions)
    ]
    time = torch.linspace(0.2, 0.9, len(locations), device=device, dtype=dtype).reshape(-1, 1)
    time.requires_grad_()
    scale = torch.tensor(0.8, device=device, dtype=dtype, requires_grad=True)

    def smooth(columns: list[torch.Tensor], clock: torch.Tensor) -> torch.Tensor:
        value = clock.mul(0.2).exp()
        phase = clock
        for i, x in enumerate(columns):
            value = value * (1 + x + x.square())
            phase = phase + (i + 1) * x
        return scale * (value + phase.sin())

    def target(
        columns: list[torch.Tensor], clock: torch.Tensor, axis: int, side: int
    ) -> torch.Tensor:
        derivative = clock.mul(0.2).exp()
        phase = clock
        for i, x in enumerate(columns):
            derivative = derivative * (1 + 2 * x if i == axis else 1 + x + x.square())
            phase = phase + (i + 1) * x
        derivative = scale * (derivative + (axis + 1) * phase.cos())
        operator = (axes[axis].lower, axes[axis].upper)[side]
        return (
            operator.alpha * smooth(columns, clock)
            + operator.beta * (-1 if side == 0 else 1) * derivative
        )

    targets = tuple(
        (partial(target, axis=i, side=0), partial(target, axis=i, side=1))
        for i in range(dimensions)
    )

    def initial(columns: list[torch.Tensor], clock: torch.Tensor) -> TensorOutputs:
        return {"u": smooth(columns, clock)}

    recipe = BoxRobinInitial(BoxRobinProjection(axes), (0.2, 0.9), initial, targets, "u")
    expanded = ExpandedRobinLifting(axes, targets, paired_time)
    initial_time = torch.full_like(time, 0.2)
    reference = recipe._lifting(dimensions, spatial, time, initial_time)["u"]
    actual = expanded.difference(spatial, time, initial_time)
    tolerance = 3e-4 if dtype == torch.float32 else 4e-11
    torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
    coordinates = Coordinates(dict(zip("xyz", spatial)), time)
    equation, expected = (
        Equation({"u": actual}, coordinates),
        Equation({"u": reference}, coordinates),
    )
    for name in (*"xyz"[:dimensions], "t"):
        for order in (1, 2):
            torch.testing.assert_close(
                equation.diff("u", name, order),
                expected.diff("u", name, order),
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
    control = expected.dt("u") - 0.1 * expected.laplacian("u")
    torch.testing.assert_close(
        torch.autograd.grad(residual.square().mean(), scale)[0],
        torch.autograd.grad(control.square().mean(), scale)[0],
        atol=tolerance,
        rtol=tolerance,
    )
    at_initial = expanded.difference(spatial, initial_time, initial_time)
    # Separate evaluations/differentiations may cancel only to rounding accuracy.
    torch.testing.assert_close(
        at_initial, torch.zeros_like(time), atol=4 * torch.finfo(dtype).eps, rtol=0
    )
    for axis, x in enumerate(spatial):
        derivative = equation.diff("u", "xyz"[axis])
        for side, operator in enumerate((axes[axis].lower, axes[axis].upper)):
            mask = x[:, 0] == axes[axis].bounds[side]
            trace = operator.alpha * actual + operator.beta * (-1 if side == 0 else 1) * derivative
            data = targets[axis][side](spatial, time) - targets[axis][side](spatial, initial_time)
            torch.testing.assert_close(trace[mask], data[mask], atol=tolerance, rtol=tolerance)
