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
    y_interval: Sequence[float] | None = None,
    y_points: int | None = None,
    z_interval: Sequence[float] | None = None,
    z_points: int | None = None,
) -> PointCloudData:
    """Generate a Cartesian 1D/2D/3D spatial grid crossed with time.

    Spatial columns share point ordering (indexing="ij", C-order flattening).
    """
    axes = [("x", x_interval, x_points)]
    for name, bounds, count in (("y", y_interval, y_points), ("z", z_interval, z_points)):
        if (bounds is None) != (count is None):
            raise ValueError(f"{name}_interval and {name}_points must be specified together")
        if bounds is not None and count is not None:
            if name == "z" and y_interval is None:
                raise ValueError("z requires a y axis")
            axes.append((name, bounds, count))
    for name, bounds in [(f"{name}_interval", bounds) for name, bounds, _ in axes] + [
        ("t_interval", t_interval)
    ]:
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"{name} must contain two finite, increasing bounds")
    for name, count in [(f"{name}_points", count) for name, _, count in axes] + [
        ("t_points", t_points)
    ]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    grids = np.meshgrid(
        *(np.linspace(bounds[0], bounds[1], count) for _, bounds, count in axes), indexing="ij"
    )
    spatial = [grid.reshape(-1, 1) for grid in grids]
    time: NDArray[np.float64] = np.linspace(t_interval[0], t_interval[1], t_points).reshape(-1, 1)
    return PointCloudData(spatial=spatial, time=time, solution={})
