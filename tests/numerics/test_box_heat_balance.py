"""Volume and face diagnostics detect drift independently of exact boundary data."""

from functools import partial

import pytest
import torch

from benchmarks.box_heat_balance import box_heat_balance
from benchmarks.cases import CASES, BoxNeumannHeatCase
from pinnstorch.api import Coordinates
from pinnstorch.runner.problem import load_problem


@pytest.mark.parametrize(
    "name", ["heat_neumann2d", "heat_neumann3d", "heat_robin2d", "heat_robin3d"]
)
def test_box_balance_analytical_and_added_drift(name: str) -> None:
    case = CASES[name]
    assert isinstance(case, BoxNeumannHeatCase)
    problem = load_problem(case.source)

    def field(spatial: list[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        mode = torch.exp(-0.1 * case.dimensions * case.frequency**2 * time)
        background = torch.ones_like(time)
        for x in spatial:
            mode = mode * torch.cos(case.frequency * (x - case.shift))
            background = background + time * x
        return background + mode

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
    operators = tuple(((case.alpha, 1.0),) * 2 for _ in range(case.dimensions))
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
    if "robin" in name:
        refined = box_heat_balance(field, **{**kwargs, "space_points": 33})
        assert refined["heat_balance_rate_rmse"] < metrics["heat_balance_rate_rmse"] / 3.5
