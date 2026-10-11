"""Batch lengths must agree with iteration and retain aligned constraint targets."""

import pytest
import torch

from pinnstorch.contracts import SampleIndex, SamplerBatch
from pinnstorch.data import PINNDataLoader


class Dataset:
    def __init__(self, size: int) -> None:
        self.values = torch.arange(size, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: SampleIndex) -> SamplerBatch:
        values = self.values[index]
        return [values], values + 10, {"mask": values + 20}


@pytest.mark.parametrize(
    "size,batch,drop,expected",
    [
        (6, 3, False, [3, 3]),
        (7, 3, False, [3, 3, 1]),
        (7, 3, True, [3, 3]),
        (2, 3, True, []),
        (2, 3, False, [2]),
        (0, None, False, []),
        (7, None, False, [7]),
    ],
)
def test_batch_sizes(size: int, batch: int | None, drop: bool, expected: list[int]) -> None:
    loader = PINNDataLoader(Dataset(size), batch_size=batch, ignore=drop)
    batches = list(loader)
    assert len(loader) == len(batches) == len(expected)
    assert [len(columns[0]) for columns, _, _ in batches] == expected
    for columns, time, targets in batches:
        torch.testing.assert_close(time, columns[0] + 10)
        torch.testing.assert_close(targets["mask"], columns[0] + 20)


@pytest.mark.parametrize("batch", [0, -1, True, 2.5])
def test_invalid_batch_size(batch: object) -> None:
    with pytest.raises(ValueError):
        PINNDataLoader(Dataset(3), batch_size=batch)


def test_shuffle_preserves_alignment_and_changes_each_iteration() -> None:
    torch.manual_seed(123)
    loader = PINNDataLoader(Dataset(20), shuffle=True)
    first = next(iter(loader))
    second = next(iter(loader))
    assert not torch.equal(first[0][0], second[0][0])
    for columns, time, targets in (first, second):
        torch.testing.assert_close(
            columns[0].sort(dim=0).values, torch.arange(20).reshape(-1, 1).float()
        )
        torch.testing.assert_close(time, columns[0] + 10)
        torch.testing.assert_close(targets["mask"], columns[0] + 20)
