"""Shared vectorized storage and slicing for condition datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import torch

from pinnstorch.contracts import (
    ArrayOutputs,
    FloatArray,
    LossAccumulator,
    SampleIndex,
    SamplerBatch,
    SamplerFunctions,
    TensorOutputs,
)


class SamplerBase(ABC):
    def __init__(self) -> None:
        self.spatial_domain_sampled: list[torch.Tensor] = []
        self.time_domain_sampled = torch.empty(0, 1)
        self.solution_sampled: list[torch.Tensor] = []
        self.solution_names: list[str] = []
        self.functions: SamplerFunctions

    @staticmethod
    def convert_to_tensor(arrays: Sequence[FloatArray]) -> list[torch.Tensor]:
        return [torch.as_tensor(array, dtype=torch.float32) for array in arrays]

    def store(
        self, spatial: FloatArray, time: FloatArray, values: ArrayOutputs, count: int | None = None
    ) -> None:
        if count is not None:
            if count > len(spatial):
                raise ValueError(
                    f"Requested {count} samples, but only {len(spatial)} points exist"
                )
            indices = np.random.choice(len(spatial), count, replace=False)
            spatial, time = spatial[indices], time[indices]
            values = {name: value[indices] for name, value in values.items()}
        columns = self.convert_to_tensor([spatial, time])
        self.spatial_domain_sampled = [columns[0][:, i : i + 1] for i in range(spatial.shape[1])]
        self.time_domain_sampled = columns[1]
        self.solution_names = list(values)
        self.solution_sampled = self.convert_to_tensor(list(values.values()))

    def loss_fn(
        self, inputs: SamplerBatch, loss: LossAccumulator, functions: SamplerFunctions
    ) -> tuple[LossAccumulator, TensorOutputs]:
        self.functions = functions
        return self._loss_fn(inputs, loss)

    @abstractmethod
    def _loss_fn(
        self, inputs: SamplerBatch, loss: LossAccumulator
    ) -> tuple[LossAccumulator, TensorOutputs]: ...

    def __len__(self) -> int:
        return len(self.time_domain_sampled)

    def __getitem__(self, index: SampleIndex) -> SamplerBatch:
        spatial = [column[index] for column in self.spatial_domain_sampled]
        time = self.time_domain_sampled[index]
        values = {
            name: column[index] for name, column in zip(self.solution_names, self.solution_sampled)
        }
        return spatial, time, values
