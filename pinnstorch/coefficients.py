"""Prescribed spatial fields with differentiable Cartesian interpolation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from pinnstorch.api import Coordinates


class RegularGridCoefficient(nn.Module):
    """Multilinear interpolation of a fixed 1D/2D/3D coefficient table."""

    values: torch.Tensor
    _strides: torch.Tensor
    _corner_bits: torch.Tensor
    _corner_offsets: torch.Tensor

    def __init__(self, axes: Mapping[str, torch.Tensor], values: torch.Tensor) -> None:
        super().__init__()
        self.axis_names = tuple(axes)
        self._validate(axes, values)
        self.signature = self._signature(axes, values)
        self.coverage = {
            name: (float(vector[0]), float(vector[-1])) for name, vector in axes.items()
        }
        self._validate({name: vector.float() for name, vector in axes.items()}, values.float())
        for name, vector in axes.items():
            self.register_buffer(name, vector.detach().clone().float())
        self.register_buffer("values", values.detach().clone().float())
        shape = tuple(len(vector) for vector in axes.values())
        strides = tuple(int(np.prod(shape[i + 1 :])) for i in range(len(shape)))
        corners = tuple(product((0, 1), repeat=len(shape)))
        offsets = [sum(bit * stride for bit, stride in zip(corner, strides)) for corner in corners]
        self.register_buffer("_strides", torch.tensor(strides, dtype=torch.long), persistent=False)
        self.register_buffer(
            "_corner_bits", torch.tensor(corners, dtype=torch.bool), persistent=False
        )
        self.register_buffer(
            "_corner_offsets", torch.tensor(offsets, dtype=torch.long), persistent=False
        )

    @staticmethod
    def _validate(axes: Mapping[str, torch.Tensor], values: torch.Tensor) -> None:
        if not axes or any(name not in ("x", "y", "z") for name in axes):
            raise ValueError("Coefficient axes must be a nonempty subset of x/y/z")
        for name, vector in axes.items():
            if (
                vector.ndim != 1
                or len(vector) < 2
                or vector.is_complex()
                or vector.dtype == torch.bool
                or not torch.isfinite(vector).all()
            ):
                raise ValueError(
                    f"Coefficient axis '{name}' must be a finite vector of length >= 2"
                )
            if not torch.all(vector[1:] > vector[:-1]):
                raise ValueError(f"Coefficient axis '{name}' must be strictly increasing")
        if tuple(values.shape) != tuple(len(vector) for vector in axes.values()):
            raise ValueError("Coefficient values shape must match axis lengths in axis order")
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError("Coefficient values must be finite floating-point numbers")

    @staticmethod
    def _signature(axes: Mapping[str, torch.Tensor], values: torch.Tensor) -> str:
        digest = hashlib.sha256()
        for name, tensor in [*axes.items(), ("values", values)]:
            digest.update(name.encode())
            array = tensor.detach().cpu().double().numpy().astype("<f8")
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    @classmethod
    def load(cls, path: Path) -> RegularGridCoefficient:
        source = np.load(path, allow_pickle=False)
        if not isinstance(source, np.lib.npyio.NpzFile):
            raise ValueError("Coefficient file must be an NPZ archive")
        with source:
            names = tuple(name for name in ("x", "y", "z") if name in source)
            if set(source.files) != {*names, "values"}:
                raise ValueError(
                    "Coefficient NPZ must contain spatial axis vectors and values only"
                )
            arrays = {name: np.asarray(source[name]) for name in (*names, "values")}
        if any(array.dtype.kind not in "iuf" for array in arrays.values()):
            raise ValueError("Coefficient NPZ must contain real numeric arrays")
        return cls(
            {name: torch.as_tensor(arrays[name], dtype=torch.float64) for name in names},
            torch.as_tensor(arrays["values"], dtype=torch.float64),
        )

    def bounds(self) -> dict[str, tuple[float, float]]:
        return dict(self.coverage)

    def forward(self, coordinates: Coordinates) -> torch.Tensor:
        cells = [
            self._cell(getattr(self, name), coordinates.variable(name)) for name in self.axis_names
        ]
        lower = torch.cat([cell[0] for cell in cells], dim=1)
        base = lower.mul(self._strides).sum(dim=1, keepdim=True)
        indices = base.add(self._corner_offsets.unsqueeze(0))
        weights: torch.Tensor = coordinates.t.new_ones(
            (len(coordinates.t), len(self._corner_offsets))
        )
        for axis, (_, fraction) in enumerate(cells):
            weights = weights * torch.where(
                self._corner_bits[:, axis].unsqueeze(0), fraction, 1 - fraction
            )
        samples = (
            self.values.reshape(-1)[indices.reshape(-1)]
            .reshape_as(weights)
            .to(coordinates.t.dtype)
        )
        return cast(torch.Tensor, (samples * weights).sum(dim=1, keepdim=True))

    @staticmethod
    def _cell(vector: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if vector.device != query.device:
            raise ValueError("Coefficient buffers and coordinates must be on the same device")
        # Bound checks happen once against the task domain, not with a GPU sync per batch.
        lower = cast(
            torch.Tensor, torch.searchsorted(vector, query.detach().contiguous(), right=True) - 1
        )
        lower = lower.clamp(0, len(vector) - 2)
        fraction = (query - vector[lower]) / (vector[lower + 1] - vector[lower])
        return lower, cast(torch.Tensor, fraction)
