from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, List, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from pinnstorch.contracts import Bounds


class FCN(nn.Module):
    """A simple fully-connected neural net for solving equations.

    In this model, lower and upper bound will be used for normalization of input data
    """

    output_names: Sequence[str]

    lb: torch.Tensor
    ub: torch.Tensor

    def __init__(
        self,
        layers: Sequence[int],
        lb: Bounds,
        ub: Bounds,
        output_names: Sequence[str],
    ) -> None:
        """Initialize a `FCN` module.

        :param layers: The list indicating number of neurons in each layer.
        :param lb: Lower bound for the inputs.
        :param ub: Upper bound for the inputs.
        :param output_names: Names of outputs of net.
        """
        super().__init__()

        self.model = self._build_layers(layers)
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32, requires_grad=False))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32, requires_grad=False))
        self.output_names = output_names

    def _build_layers(self, layers: Sequence[int]) -> nn.Sequential:
        """Initialize the layers of the neural network.

        :param layers: The list indicating number of neurons in each layer.
        :return: The initialized neural network.
        """

        initializer = nn.init.xavier_uniform_
        net = nn.Sequential()

        input_layer = nn.Linear(layers[0], layers[1])
        initializer(input_layer.weight)

        net.add_module("input", input_layer)
        net.add_module("activation_1", nn.Tanh())

        for i in range(1, len(layers) - 2):
            hidden_layer = nn.Linear(layers[i], layers[i + 1])
            initializer(hidden_layer.weight)
            net.add_module(f"hidden_{i + 1}", hidden_layer)
            net.add_module(f"activation_{i + 1}", nn.Tanh())

        output_layer = nn.Linear(layers[-2], layers[-1])
        initializer(output_layer.weight)
        net.add_module("output", output_layer)
        return net

    def forward(self, spatial: List[torch.Tensor], time: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a single forward pass through the network.

        :param spatial: List of input spatial tensors.
        :param time: Input tensor representing time.
        :return: A tensor of solutions.
        """

        features = torch.cat([*spatial, time], dim=1)
        normalized = 2.0 * (features - self.lb) / (self.ub - self.lb) - 1.0
        outputs = cast(torch.Tensor, self.model(normalized))
        return {name: outputs[:, i : i + 1] for i, name in enumerate(self.output_names)}
