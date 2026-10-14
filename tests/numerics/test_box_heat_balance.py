"""Volume and face diagnostics detect drift independently of exact boundary data."""

from functools import partial

import pytest
import torch

from benchmarks.box_heat_balance import box_heat_balance
from benchmarks.cases import CASES, BoxNeumannHeatCase
from pinnstorch.api import Coordinates
from pinnstorch.runner.problem import load_problem


@pytest.mark.parametrize(
    "name", ["heat_neumann2d", "heat_neumann3d", "heat_robin2d", "heat_robin3d", "heat_mixed3d"]
)
def test_box_balance_analytical_and_added_drift(name: str) -> None:
    case = CASES[name]
    assert isinstance(case, BoxNeumannHeatCase)
    problem = load_problem(case.source)

    def field(spatial: list[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        mode = torch.ones_like(time)
        background = torch.ones_like(time)
        decay = 0.0
        for axis, x in zip("xyz", spatial):
            definition = case.mode_for(axis)
            mode = mode * torch.cos(definition.frequency * (x - definition.shift))
            background = background + time * x
            decay += definition.frequency**2
        return background + mode * torch.exp(-0.1 * decay * time)

    def target(spatial: list[torch.Tensor], time: torch.Tensor, axis: str) -> torch.Tensor:
        return problem.boundary_values[f"wall_{axis}"](
            Coordinates(dict(zip("xyz", spatial)), time)
        )

    def source(spatial: list[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        value = torch.zeros_like(time)
        for x in spatial:
            value = value + x
        return value

    def wrong(spatial: list[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        return field(spatial, time) + time * 0.1

    targets = tuple((partial(target, axis=axis),) * 2 for axis in "xyz"[: case.dimensions])
    operators = tuple(((case.mode_for(axis).alpha, 1.0),) * 2 for axis in "xyz"[: case.dimensions])
    kwargs = dict(
        targets=targets,
        source=source,
        bounds=((0.0, 1.0),) * case.dimensions,
        time_interval=(0.0, 0.5),
        operators=operators,
        device="cpu",
        space_points=17,
        time_points=21,
        batch_size=701,
    )
    metrics = box_heat_balance(field, **kwargs)
    assert metrics["boundary_operator_rmse"] < 1e-5
    assert metrics["heat_balance_rate_rmse"] < 0.002
    assert metrics["heat_balance_accumulated_max_error"] < 0.001
    incorrect = box_heat_balance(wrong, **kwargs)
    assert incorrect["heat_balance_rate_rmse"] > 0.09
    assert incorrect["heat_balance_accumulated_max_error"] > 0.045
    gaussian = box_heat_balance(field, **{**kwargs, "quadrature": "gauss_legendre"})
    assert gaussian["heat_balance_rate_rmse"] < 2e-6
    assert (
        abs(
            gaussian["initial_heat_content"]
            - box_heat_balance(
                field, **{**kwargs, "quadrature": "gauss_legendre", "space_points": 12}
            )["initial_heat_content"]
        )
        < 2e-6
    )
    if "robin" in name:
        refined = box_heat_balance(field, **{**kwargs, "space_points": 33})
        assert refined["heat_balance_rate_rmse"] < metrics["heat_balance_rate_rmse"] / 3.5


@pytest.mark.parametrize("quadrature", ["trapezoid", "gauss_legendre"])
def test_balance_on_shifted_unequal_box_intervals(quadrature: str) -> None:
    import math

    bounds = ((-0.4, 0.7), (1, 3), (-1, 2.5))

    def source(spatial: list[torch.Tensor], clock: torch.Tensor) -> torch.Tensor:
        value = torch.zeros_like(clock)
        for x in spatial:
            value = value + x
        return value

    def field(spatial: list[torch.Tensor], clock: torch.Tensor) -> torch.Tensor:
        return 1 + clock * source(spatial, clock)

    def target(spatial: list[torch.Tensor], clock: torch.Tensor, normal: int) -> torch.Tensor:
        return normal * clock

    targets = ((partial(target, normal=-1), partial(target, normal=1)),) * 3
    result = box_heat_balance(
        field,
        targets,
        source,
        bounds,
        (0, 0.3),
        (((0, 1), (0, 1)),) * 3,
        "cpu",
        space_points=7,
        time_points=7,
        quadrature=quadrature,
    )
    volume = math.prod(upper - lower for lower, upper in bounds)
    expected = volume * (1 + 0.3 * sum((upper + lower) / 2 for lower, upper in bounds))
    assert result["initial_heat_content"] == pytest.approx(volume, abs=2e-6)
    assert result["final_heat_content"] == pytest.approx(expected, abs=3e-6)
    assert result["heat_balance_rate_rmse"] < 3e-6
    assert result["heat_balance_accumulated_max_error"] < 3e-6
