"""Sample initial values from the problem callback."""

from __future__ import annotations

import numpy as np

from pinnstorch.contracts import InitialConditions, LossAccumulator, SamplerBatch, TensorOutputs

from ..mesh.mesh import MeshBase
from .sampler_base import SamplerBase


class InitialCondition(SamplerBase):
    def __init__(
        self,
        mesh: MeshBase,
        num_sample: int | None = None,
        solution: list[str] | None = None,
        initial_fun: InitialConditions | None = None,
    ) -> None:
        super().__init__()
        if not solution:
            raise ValueError("InitialCondition must specify solution names")
        spatial, time, values = mesh.on_initial_boundary([] if initial_fun else solution)
        if initial_fun is not None:
            values = initial_fun(spatial)
        if not isinstance(values, dict):
            raise ValueError("initial_conditions must return a dictionary of arrays")
        for name in solution:
            if name not in values:
                raise ValueError(f"initial_conditions is missing solution '{name}'")
            value = np.asarray(values[name])
            if value.shape != (len(spatial), 1) or not np.isfinite(value).all():
                raise ValueError(f"initial_conditions['{name}'] must be a finite (N, 1) array")
        self.store(spatial, time, {name: values[name] for name in solution}, num_sample)

    def _loss_fn(
        self, inputs: SamplerBatch, loss: LossAccumulator
    ) -> tuple[LossAccumulator, TensorOutputs]:
        spatial, time, targets = inputs
        outputs = self.functions["forward"](spatial, time)
        return self.functions["loss_fn"](loss, outputs, targets, keys=self.solution_names), outputs
