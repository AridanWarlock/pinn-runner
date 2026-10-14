"""Independent separable identities, edge operators and PDE parameter gradients."""

import itertools
import os
from functools import partial

import pytest
import torch

from pinnstorch.api import Coordinates, Equation
from pinnstorch.contracts import TensorOutputs
from pinnstorch.training.robin_box import BoxRobinInitial, BoxRobinProjection
from pinnstorch.training.robin_projection import IntervalRobinProjection, RobinOperator


def polynomial(x: torch.Tensor) -> torch.Tensor:
    return 1 + x + x.square()


def polynomial_derivative(x: torch.Tensor) -> torch.Tensor:
    return 1 + 2 * x


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_box_projection_independent_product_and_training_gradients(
    dimensions: int, dtype: torch.dtype
) -> None:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    intervals = tuple(
        IntervalRobinProjection((-0.4 + i, 1.2 + i), RobinOperator(i, 0.7), RobinOperator(2, -1.1))
        for i in range(dimensions)
    )
    projection = BoxRobinProjection(intervals)
    points = list(
        itertools.product(*[(a.bounds[0], sum(a.bounds) / 2, a.bounds[1]) for a in intervals])
    )
    spatial = [
        torch.tensor([[p[i]] for p in points], dtype=dtype, device=device, requires_grad=True)
        for i in range(dimensions)
    ]
    time = torch.full_like(spatial[0], 0.37, requires_grad=True)
    weight = torch.tensor(0.63, dtype=dtype, device=device, requires_grad=True)
    calls = 0

    def field(columns: list[torch.Tensor], clock: torch.Tensor) -> TensorOutputs:
        nonlocal calls
        calls += 1
        value = weight * clock.exp()
        for column in columns:
            value = value * polynomial(column)
        return {"u": value}

    actual = projection.apply(field(spatial, time), field, spatial, time)["u"]
    assert calls == 3**dimensions
    reference = weight * time.exp()
    for x, interval in zip(spatial, intervals):
        basis = interval.basis(x)
        correction = polynomial(x)
        for side, operator in enumerate((interval.lower, interval.upper)):
            endpoint = torch.full_like(x, interval.bounds[side])
            normal = -1 if side == 0 else 1
            trace = operator.alpha * polynomial(
                endpoint
            ) + operator.beta * normal * polynomial_derivative(endpoint)
            correction = correction - basis[side] * trace
        reference = reference * correction
    names = tuple("xyz"[:dimensions])
    coordinates = Coordinates(dict(zip(names, spatial)), time)
    equation = Equation({"u": actual}, coordinates)
    expected = Equation({"u": reference}, coordinates)
    tolerance = 5e-4 if dtype == torch.float32 else 5e-11
    torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
    for name, x, interval in zip(names, spatial, intervals):
        for side, operator in enumerate((interval.lower, interval.upper)):
            mask = x[:, 0] == interval.bounds[side]
            normal = -1 if side == 0 else 1
            trace = operator.alpha * actual + operator.beta * normal * equation.diff("u", name)
            torch.testing.assert_close(
                trace[mask], torch.zeros_like(trace[mask]), rtol=0, atol=tolerance
            )
        for order in (1, 2):
            torch.testing.assert_close(
                equation.diff("u", name, order),
                expected.diff("u", name, order),
                rtol=tolerance,
                atol=tolerance,
            )
    mixed = equation.diff(equation.diff("u", "x"), "y")
    torch.testing.assert_close(
        mixed, expected.diff(expected.diff("u", "x"), "y"), rtol=tolerance, atol=tolerance
    )
    residual = equation.dt("u") - 0.1 * sum(equation.diff("u", name, 2) for name in names)
    reference_residual = expected.dt("u") - 0.1 * sum(
        expected.diff("u", name, 2) for name in names
    )
    torch.testing.assert_close(
        torch.autograd.grad(residual.square().mean(), weight)[0],
        torch.autograd.grad(reference_residual.square().mean(), weight)[0],
        rtol=tolerance,
        atol=tolerance,
    )
    calls = 0
    with torch.no_grad():
        outputs = field([x.detach() for x in spatial], time.detach())
        predicted = projection.apply(outputs, field, [x.detach() for x in spatial], time.detach())[
            "u"
        ]
        assert not predicted.requires_grad
        torch.testing.assert_close(predicted, actual, rtol=0, atol=0)


@pytest.mark.parametrize("dimensions", [2, 3])
def test_box_initial_time_lifting_and_every_edge_operator(dimensions: int) -> None:
    dtype = torch.float64
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    axes = tuple(
        IntervalRobinProjection((0, 1), RobinOperator(i, 1), RobinOperator(i + 1, 0.5))
        for i in range(dimensions)
    )
    projection = BoxRobinProjection(axes)
    points = list(itertools.product((0, 0.3, 1), repeat=dimensions))
    spatial = [
        torch.tensor([[p[i]] for p in points], dtype=dtype, device=device, requires_grad=True)
        for i in range(dimensions)
    ]
    time = torch.full_like(spatial[0], 0.7, requires_grad=True)

    def initial(columns: list[torch.Tensor], clock: torch.Tensor) -> TensorOutputs:
        value = torch.ones_like(clock)
        for x in columns:
            value = value * polynomial(x)
        return {"u": value}

    def target(
        columns: list[torch.Tensor], clock: torch.Tensor, axis: int, side: int
    ) -> torch.Tensor:
        operator = (axes[axis].lower, axes[axis].upper)[side]
        normal = -1 if side == 0 else 1
        value = (
            operator.alpha * polynomial(columns[axis])
            + normal * operator.beta * polynomial_derivative(columns[axis])
        ) * (1 + clock)
        for i, x in enumerate(columns):
            if i != axis:
                value = value * polynomial(x)
        return value

    targets = tuple(
        tuple(partial(target, axis=i, side=s) for s in (0, 1)) for i in range(dimensions)
    )
    recipe = BoxRobinInitial(projection, (0, 1), initial, targets, "u")

    def network(columns: list[torch.Tensor], clock: torch.Tensor) -> TensorOutputs:
        return {"u": sum(x.sin() for x in columns) + clock.cos()}

    bound = recipe.bind(network)
    actual = bound(network(spatial, time), *spatial, time)
    coordinates = Coordinates(dict(zip("xyz", spatial)), time)
    equation = Equation(actual, coordinates)
    for axis, name in enumerate("xyz"[:dimensions]):
        for side, operator in enumerate((axes[axis].lower, axes[axis].upper)):
            mask = spatial[axis][:, 0] == side
            trace = operator.alpha * actual["u"] + (
                -1 if side == 0 else 1
            ) * operator.beta * equation.diff("u", name)
            torch.testing.assert_close(
                trace[mask], targets[axis][side](spatial, time)[mask], rtol=1e-11, atol=1e-11
            )
    initial_time = torch.zeros_like(time)
    torch.testing.assert_close(
        bound(network(spatial, initial_time), *spatial, initial_time)["u"],
        initial(spatial, initial_time)["u"],
        rtol=0,
        atol=0,
    )
