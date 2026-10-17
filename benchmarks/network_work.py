"""Count actual FCN evaluations separately from timed/profiler measurements."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from pinnstorch.models.net.neural_net import FCN
from pinnstorch.models.net.spatial_features import FeatureFCN
from pinnstorch.models.pinn_module import PINNModule


@dataclass(frozen=True)
class NetworkWork:
    calls: int
    rows: int


def count_network_work(model: PINNModule, operation: Callable[[], object]) -> NetworkWork:
    network = model.net.network if isinstance(model.net, FeatureFCN) else model.net
    if not isinstance(network, FCN):
        raise ValueError("Network work counter requires an FCN")
    calls, rows = 0, 0

    def count(_module: nn.Module, inputs: tuple[object, ...]) -> None:
        nonlocal calls, rows
        features = cast(torch.Tensor, inputs[0])
        calls += 1
        rows += len(features)

    # Bound FCN.forward calls bypass outer hooks; every evaluation calls this Sequential.
    handle = network.model.register_forward_pre_hook(count)
    try:
        operation()
    finally:
        handle.remove()
    return NetworkWork(calls, rows)
