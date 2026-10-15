"""Named condition datasets and their batch loaders."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lightning import LightningDataModule

from pinnstorch.contracts import SamplerLoss

from .dataloader.dataloader import PINNDataLoader
from .sampler.mesh_sampler import MeshSampler
from .sampler.sampler_base import SamplerBase


class PINNDataModule(LightningDataModule):
    """Keep condition identities stable and preserve coordinates/targets in batching."""

    def __init__(
        self,
        train_datasets: Sequence[SamplerBase],
        val_dataset: SamplerBase | None,
        pred_dataset: SamplerBase | None = None,
        batch_size: int | None = None,
        condition_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.train_datasets = list(train_datasets)
        self.val_dataset, self.pred_dataset, self.batch_size = (
            val_dataset,
            pred_dataset,
            batch_size,
        )
        self.condition_names = tuple(
            condition_names or (f"condition_{i}" for i in range(len(train_datasets)))
        )
        if len(self.condition_names) != len(train_datasets) or len(
            set(self.condition_names)
        ) != len(self.condition_names):
            raise ValueError("Condition names must be unique and match the training datasets")
        self.function_mapping: dict[str, SamplerLoss] = {}
        self._training_loaders: dict[str, PINNDataLoader] | None = None
        self.solution_names: list[str] | None = None
        self.loss_fn: SamplerLoss

    def setup(self, stage: str | None = None) -> None:
        if self._training_loaders is None:
            self._training_loaders = {}
            for name, dataset in zip(self.condition_names, self.train_datasets):
                self._training_loaders[name] = PINNDataLoader(
                    dataset, self.batch_size, shuffle=True
                )
                self.function_mapping[name] = dataset.loss_fn

    def train_dataloader(self) -> dict[str, PINNDataLoader]:
        # Refresh before Lightning creates/prefetches the epoch iterator.
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            self.refresh_collocation(trainer.current_epoch)
        self.setup("fit")
        assert self._training_loaders is not None
        return self._training_loaders

    def refresh_collocation(self, epoch: int) -> None:
        for dataset in self.train_datasets:
            if isinstance(dataset, MeshSampler):
                dataset.refresh(epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            name: {
                "signature": dataset.sampling_signature(),
                "epoch": dataset.last_sampling_epoch,
                "spatial": [
                    column.detach().cpu().clone() for column in dataset.spatial_domain_sampled
                ],
                "time": dataset.time_domain_sampled.detach().cpu().clone(),
            }
            for name, dataset in zip(self.condition_names, self.train_datasets)
            if isinstance(dataset, MeshSampler) and dataset.sampling_policy is not None
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for name, dataset in zip(self.condition_names, self.train_datasets):
            if name not in state_dict:
                continue
            saved = state_dict[name]
            if (
                not isinstance(dataset, MeshSampler)
                or dataset.sampling_policy is None
                or tuple(saved["signature"]) != dataset.sampling_signature()
            ):
                raise ValueError("Checkpoint sampling recipe differs from current task")
            dataset.spatial_domain_sampled = [column.clone() for column in saved["spatial"]]
            dataset.time_domain_sampled = saved["time"].clone()
            dataset.last_sampling_epoch = int(saved["epoch"])

    def val_dataloader(self) -> PINNDataLoader | list[PINNDataLoader]:
        if self.val_dataset is None:
            return []
        self.loss_fn = self.val_dataset.loss_fn
        self.solution_names = self.val_dataset.solution_names
        return PINNDataLoader(self.val_dataset, self.batch_size)

    def predict_dataloader(self) -> PINNDataLoader:
        dataset = self.pred_dataset if self.pred_dataset is not None else self.val_dataset
        if dataset is None:
            raise ValueError("Prediction requires a prediction dataset")
        return PINNDataLoader(dataset, self.batch_size)
