"""Специализированное поле для k=a(1+x/b)^2 с модальными поправками."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from torch import nn

from pinnstorch.contracts import TensorOutputs

from .neural_net import FCN

InitialField = Callable[[list[torch.Tensor], torch.Tensor], TensorOutputs]


class QuadraticModalFCN(nn.Module):
    def __init__(
        self,
        hidden_layers: Sequence[int],
        bounds: tuple[tuple[float, float], ...],
        initial: InitialField,
        source_sha256: str,
        normalization: str = "identity",
    ) -> None:
        super().__init__()
        if normalization not in ("identity", "mode_gap"):
            raise ValueError("Unknown modal coefficient normalization")
        self.normalization = normalization
        self.physical_bounds = bounds
        self.initial = initial
        self.source_sha256 = source_sha256
        self.names = tuple(f"d{n}" for n in range(2, 33))
        self.net = FCN([1, *hidden_layers, 31], [bounds[2][0]], [bounds[2][1]], self.names)
        self.register_buffer("mode_numbers", torch.arange(2, 33, dtype=torch.float32)[None, :])
        self.register_buffer("coefficient_scale", coefficient_scale(normalization).float())

    def identity(self) -> dict[str, Any]:
        return {
            "kind": "quadratic_modal_initial",
            "version": 1,
            "basis": "r^-1/2*sin(n*pi*log(r)/log1p(1/b))",
            "eigenvalues": "a/b^2*((n*pi/log1p(1/b))^2+1/4)",
            "physical_names": ["x", "a", "b", "t"],
            "physical_bounds": [list(pair) for pair in self.physical_bounds],
            "network_names": ["b"],
            "network_bounds": [list(self.physical_bounds[2])],
            "modes": list(range(2, 33)),
            "coefficient_normalization": self.normalization,
            "coefficient_scale_rule": "3/(n^2-1)" if self.normalization == "mode_gap" else "1",
            "initial_rule": "Problem.initial at t=0",
            "source_sha256": self.source_sha256,
        }

    def coefficients(self, b: torch.Tensor) -> torch.Tensor:
        output = self.net([], b)
        return cast(
            torch.Tensor,
            torch.cat([output[name] for name in self.names], dim=1) * self.coefficient_scale,
        )

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

    def forward(self, features: list[torch.Tensor], time: torch.Tensor) -> TensorOutputs:
        if len(features) != 3:
            raise ValueError("Quadratic modal field requires physical inputs x,a,b,t")
        for name, value, (lower, upper) in zip(
            ("x", "a", "b", "t"), [*features, time], self.physical_bounds
        ):
            tolerance = 4 * torch.finfo(torch.float32).eps * max(abs(lower), abs(upper), 1)
            if (
                not torch.isfinite(value).all()
                or (value < lower - tolerance).any()
                or (value > upper + tolerance).any()
            ):
                raise ValueError(
                    f"Quadratic modal physical input '{name}' lies outside its contract"
                )
        x, a, b = features
        phi, eigen, first = self.mode_data(x, a, b)
        envelope = torch.exp(-first * time)
        initial = self.initial(features, torch.zeros_like(time))["u"]
        corrections = self.coefficients(b) * phi * (torch.exp(-eigen * time) - envelope)
        return {"u": cast(torch.Tensor, initial * envelope + corrections.sum(dim=1, keepdim=True))}


def coefficient_scale(normalization: str) -> torch.Tensor:
    modes = torch.arange(2, 33, dtype=torch.float64)[None, :]
    return 3 / (modes.square() - 1) if normalization == "mode_gap" else torch.ones_like(modes)


def validate_modal_checkpoint(checkpoint: dict[str, Any], net: nn.Module) -> None:
    expected = net.identity() if isinstance(net, QuadraticModalFCN) else None
    saved = checkpoint.get("hyper_parameters", {}).get("field_representation")
    if saved != expected:
        raise ValueError("Checkpoint modal field representation differs from task")
    if isinstance(net, QuadraticModalFCN):
        state = checkpoint.get("state_dict", {})
        for key, expected_tensor in (
            ("net.mode_numbers", torch.arange(2, 33, dtype=torch.float64)[None, :]),
            ("net.coefficient_scale", coefficient_scale(net.normalization).float()),
            ("net.net.lb", torch.tensor([net.physical_bounds[2][0]], dtype=torch.float32)),
            ("net.net.ub", torch.tensor([net.physical_bounds[2][1]], dtype=torch.float32)),
        ):
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or not torch.equal(
                value.cpu(), expected_tensor.to(dtype=value.dtype)
            ):
                raise ValueError(
                    "Checkpoint modal basis or normalization differs from its contract"
                )
