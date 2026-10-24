"""Модальные временные поправки; исследовательский класс, не файловый backend."""

from __future__ import annotations

from typing import cast

import torch

from pinnstorch.models.net.neural_net import FCN


class QuadraticModalPrototype(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.names = tuple(f"d{n}" for n in range(2, 33))
        self.net = FCN([1, 32, 32, 32, 31], [1.0], [2.0], self.names)
        self.register_buffer("mode_numbers", torch.arange(2, 33, dtype=torch.float64)[None, :])

    def coefficients(self, b: torch.Tensor) -> torch.Tensor:
        output = self.net([], b)
        return torch.cat([output[name] for name in self.names], dim=1)

    def mode_data(
        self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = torch.log1p(1 / b)
        xi = torch.log1p(x / b) / length
        n = self.mode_numbers
        # Отражение — тождество синуса, не маска поля: на концах сохраняются
        # ненулевые физические производные, а sin(n*pi) не вносит ошибку границы.
        right = xi > 0.5
        reflected = torch.where(right, 1 - xi, xi)
        parity = torch.where(n.to(torch.int64) % 2 == 0, -1, 1)
        sine = torch.sin(torch.pi * n * reflected)
        sine = torch.where(right, parity * sine, sine)
        phi = (1 + x / b) ** -0.5 * sine
        eigen = a / b**2 * ((torch.pi * n / length) ** 2 + 0.25)
        first = a / b**2 * ((torch.pi / length) ** 2 + 0.25)
        return phi, eigen, first

    def forward(self, features: list[torch.Tensor], time: torch.Tensor) -> dict[str, torch.Tensor]:
        x, a, b = features
        phi, eigen, first = self.mode_data(x, a, b)
        envelope = torch.exp(-first * time)
        initial = torch.sin(x.mul(torch.pi)) ** 3
        corrections = self.coefficients(b) * phi * (torch.exp(-eigen * time) - envelope)
        return {"u": cast(torch.Tensor, initial * envelope + corrections.sum(dim=1, keepdim=True))}
