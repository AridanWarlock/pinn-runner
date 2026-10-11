"""Shared shapes and callable contracts for the Python side of the runner.

Array dimensions are documented here and checked at task boundaries. Python's
type checker tracks element types, but does not prove tensor or array shapes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from pinnstorch.data.domains.point_cloud import PointCloudData
    from pinnstorch.data.mesh.mesh import MeshBase

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
BoundaryFunctions: TypeAlias = Mapping[str, FieldFunction]
OptimizerFactory: TypeAlias = Callable[..., torch.optim.Optimizer]
SchedulerFactory: TypeAlias = Callable[..., torch.optim.lr_scheduler.LRScheduler]


class SamplerDataset(Protocol):
    """The interface consumed by PINNDataLoader; no inheritance is required."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: SampleIndex) -> SamplerBatch: ...


class PlotFunction(Protocol):
    def __call__(
        self,
        mesh: MeshBase,
        predictions: ArrayOutputs,
        train_datasets: Sequence[SamplerDataset],
        val_dataset: SamplerDataset | None,
    ) -> None: ...


class UserFunctions(TypedDict):
    """Required and optional entries loaded from a task's functions.py."""

    pde_fn: FieldFunction
    initial_conditions: InitialConditions
    output_fn: NotRequired[FieldFunction]
    boundary_functions: NotRequired[BoundaryFunctions]
    boundary_values: NotRequired[Mapping[str, Callable[..., torch.Tensor]]]
    plot_fn: NotRequired[PlotFunction]


class SamplerFunctions(TypedDict):
    """Runtime function registry passed to the loss samplers.

    CUDA/Runge-Kutta helpers remain dynamic until their separate type migration.
    """

    forward: FieldFunction
    pde_fn: FieldFunction
    loss_fn: Callable[..., LossAccumulator]
    extra_variables: torch.nn.ParameterDict | None
    output_fn: FieldFunction | None
    runge_kutta: Any
    net: torch.nn.Module
    jit_compile: bool
    val: NotRequired[bool]
    batch_size: NotRequired[bool]


SamplerLoss: TypeAlias = Callable[
    [SamplerBatch, LossAccumulator, SamplerFunctions],
    tuple[LossAccumulator, TensorOutputs],
]
