"""Vectorized sampling of equation residuals and prediction coordinates."""

from __future__ import annotations

from pinnstorch.contracts import LossAccumulator, SamplerBatch, TensorOutputs
from pinnstorch.training.sampling import SamplingPolicy

from ..mesh.mesh import MeshBase
from .sampler_base import SamplerBase


class MeshSampler(SamplerBase):
    def __init__(
        self,
        mesh: MeshBase,
        idx_t: int | None = None,
        num_sample: int | None = None,
        solution: list[str] | None = None,
        collection_points: list[str] | None = None,
        use_lhs: bool = True,
        sampling_policy: SamplingPolicy | None = None,
    ) -> None:
        super().__init__()
        self.sampling_policy = sampling_policy
        self.last_sampling_epoch = 0
        self.collection_points_names = collection_points or []
        if solution:
            if idx_t is not None:
                spatial, time, values = mesh.on_initial_boundary(solution, idx_t)
            else:
                spatial, time, values = mesh.flatten_mesh(solution)
            self.store(spatial, time, values, num_sample)
        else:
            spatial, time = mesh.collection_points(num_sample, use_lhs)
            self.store(spatial, time, {})

    def refresh(self, epoch: int) -> None:
        policy = self.sampling_policy
        if (
            policy is not None
            and epoch > self.last_sampling_epoch
            and epoch % policy.every_epochs == 0
        ):
            points = policy.points(epoch)
            self.store(points[:, :-1], points[:, -1:], {})
            self.last_sampling_epoch = epoch

    def _loss_fn(
        self, inputs: SamplerBatch, loss: LossAccumulator
    ) -> tuple[LossAccumulator, TensorOutputs]:
        spatial, time, targets = inputs
        outputs = self.functions["forward"](spatial, time)
        if self.collection_points_names:
            outputs = self.functions["pde_fn"](outputs, *spatial, time)
            loss = self.functions["loss_fn"](loss, outputs, keys=self.collection_points_names)
        if self.solution_names:
            loss = self.functions["loss_fn"](loss, outputs, targets, keys=self.solution_names)
        return loss, outputs
