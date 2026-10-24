"""Явное преобразование физических входов; уравнение остаётся в физических координатах."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import nn

from pinnstorch.contracts import TensorOutputs

from .neural_net import FCN


class ScaledTimeFCN(nn.Module):
    def __init__(
        self,
        hidden_layers: Sequence[int],
        input_names: tuple[str, ...],
        bounds: tuple[tuple[float, float], ...],
        output_names: Sequence[str],
        parameter: str,
        reference: float,
    ) -> None:
        super().__init__()
        self.parameter_index = input_names.index(parameter)
        self.reference = reference
        self.physical_names = input_names
        self.physical_bounds = bounds
        self.kept = tuple(i for i in range(len(input_names) - 1) if i != self.parameter_index)
        self.network_names = tuple(input_names[i] for i in self.kept) + ("s",)
        self.network_bounds = tuple(bounds[i] for i in self.kept) + (bounds[-1],)
        self.network = FCN(
            [len(self.network_names), *hidden_layers, len(output_names)],
            [pair[0] for pair in self.network_bounds],
            [pair[1] for pair in self.network_bounds],
            output_names,
        )

    def identity(self) -> dict[str, Any]:
        return {
            "kind": "parameter_scaled_time",
            "version": 1,
            "parameter": self.physical_names[self.parameter_index],
            "reference": self.reference,
            "physical_names": list(self.physical_names),
            "physical_bounds": [list(pair) for pair in self.physical_bounds],
            "network_names": list(self.network_names),
            "network_bounds": [list(pair) for pair in self.network_bounds],
        }

    def forward(self, columns: list[torch.Tensor], time: torch.Tensor) -> TensorOutputs:
        if len(columns) != len(self.physical_names) - 1:
            raise ValueError("Scaled time physical input count differs from its contract")
        for name, value, (lower, upper) in zip(
            self.physical_names, [*columns, time], self.physical_bounds
        ):
            tolerance = 4 * torch.finfo(torch.float32).eps * max(abs(lower), abs(upper), 1)
            if (
                not torch.isfinite(value).all()
                or (value < lower - tolerance).any()
                or (value > upper + tolerance).any()
            ):
                raise ValueError(f"Scaled time physical input '{name}' lies outside its contract")
        scaled = columns[self.parameter_index] / self.reference * time
        return cast(TensorOutputs, self.network([columns[i] for i in self.kept], scaled))


def validate_time_scale_checkpoint(checkpoint: dict[str, Any], net: nn.Module) -> None:
    """Проверить смысл правила и нормировку до присваивания сохранённых весов."""
    expected = net.identity() if isinstance(net, ScaledTimeFCN) else None
    saved = checkpoint.get("hyper_parameters", {}).get("input_transform")
    if saved != expected:
        raise ValueError("Checkpoint input transform differs from task")
    if isinstance(net, ScaledTimeFCN):
        state = checkpoint.get("state_dict", {})
        for name, bounds in (
            ("lb", [b[0] for b in net.network_bounds]),
            ("ub", [b[1] for b in net.network_bounds]),
        ):
            value = state.get(f"net.network.{name}")
            if not isinstance(value, torch.Tensor) or not torch.equal(
                value.cpu(), torch.tensor(bounds, dtype=torch.float32).to(dtype=value.dtype)
            ):
                raise ValueError("Checkpoint scaled time normalization differs from its contract")
