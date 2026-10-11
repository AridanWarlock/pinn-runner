from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pinnstorch.contracts import FloatArray


@dataclass
class PointCloudData:
    """A data class to hold point cloud data, including spatial points, time values, and solution
    data.

    This is an alternative class to spatial and time class that should be used along with `PointCloud` class.
    """

    spatial: list[FloatArray]
    time: FloatArray | list[FloatArray]
    solution: dict[str, FloatArray]


def create_interval_data(
    x_interval: Sequence[float],
    t_interval: Sequence[float],
    x_points: int,
    t_points: int,
) -> PointCloudData:
    """Generate 1D space-time coordinates without reference solution values."""
    for name, bounds in (("x_interval", x_interval), ("t_interval", t_interval)):
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"{name} must contain two finite, increasing bounds")
    for name, count in (("x_points", x_points), ("t_points", t_points)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    x: NDArray[np.float64] = np.linspace(x_interval[0], x_interval[1], x_points).reshape(-1, 1)
    time: NDArray[np.float64] = np.linspace(t_interval[0], t_interval[1], t_points).reshape(-1, 1)
    return PointCloudData(spatial=[x], time=time, solution={})
