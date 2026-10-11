"""Shared shapes and callable contracts for the Python side of the runner.

Array dimensions are documented here and checked at task boundaries. Python's
type checker tracks element types, but does not prove tensor or array shapes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pinnstorch.data.domains.point_cloud import PointCloudData

FloatArray: TypeAlias = NDArray[np.float32 | np.float64]
ArrayOutputs: TypeAlias = dict[str, FloatArray]
TensorOutputs: TypeAlias = dict[str, torch.Tensor]
SpatialTensors: TypeAlias = list[torch.Tensor]
# Samplers return physical/parameter feature columns, one time column, and targets.
SamplerBatch: TypeAlias = tuple[SpatialTensors, torch.Tensor, TensorOutputs]
SampleIndex: TypeAlias = int | slice | torch.Tensor
LossAccumulator: TypeAlias = torch.Tensor | float
Bounds: TypeAlias = Sequence[float] | FloatArray
InitialConditions: TypeAlias = Callable[..., ArrayOutputs]
ReadData: TypeAlias = Callable[[], "PointCloudData"]
# PDE/output functions have different numbers of coordinates in 1D/2D/3D.
FieldFunction: TypeAlias = Callable[..., TensorOutputs]
OptimizerFactory: TypeAlias = Callable[..., torch.optim.Optimizer]
SchedulerFactory: TypeAlias = Callable[..., torch.optim.lr_scheduler.LRScheduler]


class SamplerDataset(Protocol):
    """The interface consumed by PINNDataLoader; no inheritance is required."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: SampleIndex) -> SamplerBatch: ...


class SamplerFunctions(TypedDict):
    """Vectorized model and loss callbacks used by condition datasets."""

    forward: FieldFunction
    pde_fn: FieldFunction
    loss_fn: Callable[..., LossAccumulator]


SamplerLoss: TypeAlias = Callable[
    [SamplerBatch, LossAccumulator, SamplerFunctions],
    tuple[LossAccumulator, TensorOutputs],
]
