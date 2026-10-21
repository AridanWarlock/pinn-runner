"""Прототип физических часов; публичный runner и checkpoint API не меняет."""

from __future__ import annotations

from typing import cast

import torch

from pinnstorch.models.net.neural_net import FCN


class QuadraticClockPrototype(torch.nn.Module):
    """Физический код (a,b) обрабатывается сетью (x,b,s), s=a/A*t."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("reference_scale", torch.tensor(0.15, dtype=torch.float64))
        self.net = FCN([3, 32, 32, 32, 1], [0.0, 1.0, 0.0], [1.0, 2.0, 1.0], ["raw"])

    def reference_field(self, x: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        raw = self.net([x, b], s)["raw"]
        return cast(torch.Tensor, torch.sin(x.mul(torch.pi)) ** 3 + s * x * (1 - x) * raw)

    def forward(self, features: list[torch.Tensor], t: torch.Tensor) -> dict[str, torch.Tensor]:
        x, a, b = features
        s = a / self.reference_scale * t
        return {"u": self.reference_field(x, b, s)}
