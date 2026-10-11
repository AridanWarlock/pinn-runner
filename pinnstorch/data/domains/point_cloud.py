from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from pinnstorch.contracts import FloatArray


class ParameterSpec(TypedDict):
    interval: Sequence[float]
    points: int


@dataclass
class PointCloudData:
    """A data class to hold point cloud data, including spatial points, time values, and solution
    data.

    This is an alternative class to spatial and time class that should be used along with `PointCloud` class.
    """

    spatial: list[FloatArray]
    time: FloatArray | list[FloatArray]
    solution: dict[str, FloatArray]
    spatial_dimension: int | None = None
    parameter_names: tuple[str, ...] = ()


def create_interval_data(
    x_interval: Sequence[float],
    t_interval: Sequence[float],
    x_points: int,
    t_points: int,
    y_interval: Sequence[float] | None = None,
    y_points: int | None = None,
    z_interval: Sequence[float] | None = None,
    z_points: int | None = None,
    parameters: Mapping[str, ParameterSpec] | None = None,
) -> PointCloudData:
    """Generate a Cartesian space/parameter grid crossed with time.

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
    spatial_dimension = len(axes)
    for name, spec in (parameters or {}).items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or name.startswith("_")
            or not name.isascii()
            or name in ("x", "y", "z", "t")
        ):
            raise ValueError(
                "Parameter names must be ASCII identifiers without a leading underscore, distinct from x/y/z/t"
            )
        if not isinstance(spec, Mapping) or set(spec) != {"interval", "points"}:
            raise ValueError(f"Parameter '{name}' requires interval and points")
        axes.append((name, spec["interval"], spec["points"]))
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
    return PointCloudData(
        spatial=spatial,
        time=time,
        solution={},
        spatial_dimension=spatial_dimension,
        parameter_names=tuple(parameters or {}),
    )
