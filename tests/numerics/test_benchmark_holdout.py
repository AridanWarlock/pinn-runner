"""Independent residual probes must cover physical input intervals, not a unit cube."""

import numpy as np

from benchmarks.cases import CASES
from benchmarks.execution import holdout_inputs
from pinnstorch.runner.settings import TaskSettings


def test_shifted_space_time_and_parameter_holdout_covers_each_named_axis() -> None:
    config = CASES["heat2d"].config("cpu", 1, 19)
    intervals = ((-2, 3), (4, 7), (0.1, 0.5), (2, 4))
    config["domain"]["space"]["x"]["interval"] = list(intervals[0])
    config["domain"]["space"]["y"]["interval"] = list(intervals[1])
    config["domain"]["parameters"] = {
        "conductivity": {"interval": list(intervals[2]), "points": 3}
    }
    config["domain"]["time"]["interval"] = list(intervals[3])
    settings = TaskSettings.read(config)
    points = holdout_inputs(settings)
    assert settings.domain.input_names == ("x", "y", "conductivity", "t")
    assert points.shape == (2048, 4)
    for column, (lower, upper) in zip(points.T, intervals):
        assert column.min() >= lower
        assert column.max() <= upper
        assert column.min() < lower + (upper - lower) * 0.02
        assert column.max() > upper - (upper - lower) * 0.02
    np.testing.assert_array_equal(points, holdout_inputs(settings))


def test_unit_box_holdout_preserves_existing_benchmark_random_points() -> None:
    settings = TaskSettings.read(CASES["heat_neumann3d"].config("cpu", 1, 19))
    expected = np.random.default_rng(20261013).uniform(0, 1, (2048, 4))
    expected[:, -1] *= 0.5
    np.testing.assert_array_equal(holdout_inputs(settings), expected)
