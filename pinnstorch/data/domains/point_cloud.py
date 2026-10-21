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
    """Aligned Cartesian feature columns, time and optional reference fields."""

    spatial: list[FloatArray]
    time: FloatArray | list[FloatArray]
    solution: dict[str, FloatArray]
    spatial_dimension: int | None = None
    parameter_names: tuple[str, ...] = ()
    normalization_bounds: tuple[FloatArray, FloatArray] | None = None
    parameter_rows: FloatArray | None = None


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
    parameter_values: Mapping[str, Sequence[float]] | None = None,
    parameter_rows: Sequence[Sequence[float]] | None = None,
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
    axes.extend(_parameter_axes(parameters or {}))
    _validate_axes([*axes, ("t", t_interval, t_points)])
    requested_values = parameter_values or {}
    unknown = set(requested_values) - set(parameters or {})
    if unknown:
        raise ValueError(f"Unknown prediction parameters: {sorted(unknown)}")
    rows = (
        validate_parameter_rows(parameter_rows, len(parameters or {}))
        if parameter_rows is not None
        else None
    )
    if rows is not None and requested_values:
        raise ValueError("parameter_rows and independent parameter_values are mutually exclusive")
    grid_axes = axes if rows is None else axes[:spatial_dimension]
    vectors = [
        _axis_vector(name, bounds, count, requested_values) for name, bounds, count in grid_axes
    ]
    grids = np.meshgrid(*vectors, indexing="ij")
    lower = np.asarray([bounds[0] for _, bounds, _ in axes] + [t_interval[0]], dtype=np.float64)
    upper = np.asarray([bounds[1] for _, bounds, _ in axes] + [t_interval[1]], dtype=np.float64)
    spatial = [grid.reshape(-1, 1) for grid in grids]
    if rows is not None:
        physical_count = len(spatial[0])
        spatial = [np.repeat(column, len(rows), axis=0) for column in spatial]
        codes = np.tile(rows, (physical_count, 1))
        spatial.extend(codes[:, index : index + 1] for index in range(rows.shape[1]))
    time: NDArray[np.float64] = np.linspace(t_interval[0], t_interval[1], t_points).reshape(-1, 1)
    return PointCloudData(
        spatial=spatial,
        time=time,
        solution={},
        spatial_dimension=spatial_dimension,
        parameter_names=tuple(parameters or {}),
        normalization_bounds=(lower, upper),
        parameter_rows=rows,
    )


def validate_parameter_rows(rows: Sequence[Sequence[float]], width: int) -> FloatArray:
    """Validate complete codes without constructing component-wise products."""
    try:
        raw = np.asarray(rows)
    except ValueError as error:
        raise ValueError("parameter_rows must be a rectangular numeric table") from error
    if (
        width < 1
        or raw.ndim != 2
        or raw.shape[0] < 1
        or raw.shape[1] != width
        or raw.dtype.kind not in "iuf"
        or any(isinstance(value, (bool, np.bool_)) for row in rows for value in row)
    ):
        raise ValueError(f"parameter_rows must be a nonempty numeric table with {width} columns")
    result: FloatArray = np.asarray(raw, dtype=np.float64).copy()
    if not np.isfinite(result).all() or len(np.unique(result, axis=0)) != len(result):
        raise ValueError("parameter_rows must be finite and contain distinct complete codes")
    return result


AxisDefinition = tuple[str, Sequence[float], int]


def _parameter_axes(parameters: Mapping[str, ParameterSpec]) -> list[AxisDefinition]:
    axes: list[AxisDefinition] = []
    for name, spec in parameters.items():
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
    return axes


def _validate_axes(axes: Sequence[AxisDefinition]) -> None:
    for name, bounds in [(f"{name}_interval", bounds) for name, bounds, _ in axes]:
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"{name} must contain two finite, increasing bounds")
    for name, count in [(f"{name}_points", count) for name, _, count in axes]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError(f"{name} must be an integer >= 2")


def _axis_vector(
    name: str, bounds: Sequence[float], count: int, requested_values: Mapping[str, Sequence[float]]
) -> FloatArray:
    if name in requested_values:
        raw = np.asarray(requested_values[name])
        if (
            raw.ndim != 1
            or not raw.size
            or raw.dtype.kind not in "iuf"
            or any(isinstance(value, (bool, np.bool_)) for value in requested_values[name])
        ):
            raise ValueError(f"Prediction values for '{name}' must be a nonempty numeric list")
        values = raw.astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Prediction values for '{name}' must be a nonempty finite list")
        if len(np.unique(values)) != len(values):
            raise ValueError(f"Prediction values for '{name}' must be distinct")
        return values
    else:
        return np.linspace(bounds[0], bounds[1], count)
