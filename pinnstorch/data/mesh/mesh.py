"""Cartesian coordinates shared by condition samplers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
from pyDOE import lhs

from pinnstorch.contracts import ArrayOutputs, FloatArray

from ..domains.point_cloud import PointCloudData


class MeshBase:
    lb: FloatArray
    ub: FloatArray
    spatial_domain: list[FloatArray]
    time_domain: FloatArray
    spatial_domain_mesh: FloatArray
    time_domain_mesh: FloatArray
    spatial_dim: int
    parameter_names: tuple[str, ...]
    solution: ArrayOutputs

    def domain_bounds(self) -> tuple[FloatArray, FloatArray]:
        features = self.spatial_domain_mesh.reshape(-1, self.spatial_domain_mesh.shape[-1])
        inputs = np.column_stack((features, self.time_domain_mesh.reshape(-1)))
        return inputs.min(axis=0), inputs.max(axis=0)

    def on_initial_boundary(
        self, solution_names: Sequence[str], idx: int = 0
    ) -> tuple[FloatArray, FloatArray, ArrayOutputs]:
        spatial = self.spatial_domain_mesh[:, idx, :]
        time = self.time_domain_mesh[:, idx, :]
        values = {name: self.solution[name][:, idx : idx + 1] for name in solution_names}
        return spatial, time, values

    def collection_points(
        self, count: int | None, use_lhs: bool = True
    ) -> tuple[FloatArray, FloatArray]:
        if not use_lhs:
            spatial, time, _ = self.flatten_mesh(None)
            return spatial, time
        if count is None or count < 1:
            raise ValueError("LHS sampling requires a positive point count")
        inputs = cast(FloatArray, self.lb + (self.ub - self.lb) * lhs(len(self.lb), count))
        return inputs[:, :-1], inputs[:, -1:]

    def flatten_mesh(
        self, solution_names: Sequence[str] | None
    ) -> tuple[FloatArray, FloatArray, ArrayOutputs]:
        spatial = self.spatial_domain_mesh.reshape(-1, self.spatial_domain_mesh.shape[-1])
        time = self.time_domain_mesh.reshape(-1, 1)
        values = {name: self.solution[name].reshape(-1, 1) for name in solution_names or ()}
        return spatial, time, values


class PointCloud(MeshBase):
    def __init__(
        self,
        read_data_fn: Callable[[], PointCloudData],
        ub: Sequence[float] | None = None,
        lb: Sequence[float] | None = None,
    ) -> None:
        data = read_data_fn()
        self.spatial_domain = data.spatial
        if isinstance(data.time, list):
            if len(data.time) != 1:
                raise ValueError("PointCloud requires one time column")
            self.time_domain = data.time[0]
        else:
            self.time_domain = data.time
        self.solution = data.solution
        spatial_count, time_count = len(self.spatial_domain[0]), len(self.time_domain)
        self._validate_shapes(spatial_count, time_count)
        self.spatial_dim = data.spatial_dimension or len(self.spatial_domain)
        self.parameter_names = data.parameter_names
        self.input_dim = len(self.spatial_domain)
        self.spatial_domain_mesh = np.empty((spatial_count, time_count, self.input_dim))
        for index, column in enumerate(self.spatial_domain):
            self.spatial_domain_mesh[:, :, index] = column
        self.time_domain_mesh = np.broadcast_to(
            self.time_domain.reshape(1, time_count, 1), (spatial_count, time_count, 1)
        )
        if (lb is None) != (ub is None):
            raise ValueError("Both lower and upper normalization bounds must be provided")
        if lb is None or ub is None:
            self.lb, self.ub = data.normalization_bounds or self.domain_bounds()
        else:
            self.lb, self.ub = np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)

    def _validate_shapes(self, spatial_count: int, time_count: int) -> None:
        if any(column.shape != (spatial_count, 1) for column in self.spatial_domain):
            raise ValueError("Spatial and parameter columns must have the same (N, 1) shape")
        if not isinstance(self.solution, dict):
            raise ValueError("Solution must be a dictionary")
        for name, values in self.solution.items():
            if values.shape != (spatial_count, time_count):
                raise ValueError(f"Solution '{name}' does not match the coordinate grid")
