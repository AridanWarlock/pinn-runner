from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pinnstorch.contracts import SamplerBatch, SamplerDataset


class PINNDataLoader:
    """Custom DataLoader for the PINN datasets.

    This allows you to have a fast dataloader.
    """

    def __init__(
        self,
        dataset: SamplerDataset,
        batch_size: int | None = None,
        ignore: bool = False,
        shuffle: bool = False,
    ) -> None:
        """Initialize a PINNDataLoader.

        :param dataset: The dataset to load data from.
        :param batch_size: The batch size for the dataloader.
        :param ignore: Whether to ignore incomplete batches (default is False).
        :param shuffle: Whether to shuffle the dataset (default is False).
        """
        if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer or None")
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_size = len(self.dataset)
        self.shuffle = shuffle
        self.current_index = 0
        self.ignore = ignore

        if self.shuffle:
            self.indices = torch.randperm(self.dataset_size)
        else:
            self.indices = torch.arange(self.dataset_size)

    def __len__(self) -> int:
        """Get the number of batches in the dataloader.

        :return: The number of batches.
        """
        if self.dataset_size == 0:
            return 0
        if self.batch_size is None:
            return 1
        if self.ignore:
            return self.dataset_size // self.batch_size
        else:
            return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> PINNDataLoader:
        """Initialize the data loader iterator.

        :return: The data loader iterator.
        """
        self.current_index = 0
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_size)
        return self

    def __next__(self) -> SamplerBatch:
        """Generate the next batch of data.

        :return: The next batch of data.
        """
        if self.current_index >= len(self.indices):
            raise StopIteration

        # If batch_size is None, return the entire dataset as a single batch
        if self.batch_size is None:
            batch = self.dataset[self.indices] if self.shuffle else self.dataset[:]
            self.current_index += self.dataset_size
            return batch

        if self.ignore and len(self.indices) - self.current_index < self.batch_size:
            raise StopIteration

        batch_indices = self.indices[self.current_index : self.current_index + self.batch_size]
        batch = self.dataset[batch_indices]
        self.current_index += self.batch_size

        return batch
