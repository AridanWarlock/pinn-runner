"""Непрерывная PDE-потеря по модам: только проекции известных начальных данных.

Это исследовательский функционал, пока не backend публичного обучения.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from pinnstorch.models.net.quadratic_modal import InitialField


class ModalResidualObjective(nn.Module):
    """Обучаемая часть точного интеграла; постоянный хвост хранится отдельно."""

    def __init__(
        self,
        initial: InitialField,
        forms: list[float],
        scale: float = 0.15,
        time_end: float = 1.0,
        quadrature: int = 256,
    ) -> None:
        super().__init__()
        if (
            not forms
            or len(set(forms)) != len(forms)
            or not np.isfinite(forms).all()
            or min(forms) <= 0
            or not np.isfinite(scale)
            or scale <= 0
            or not np.isfinite(time_end)
            or time_end <= 0
            or isinstance(quadrature, bool)
            or quadrature < 64
        ):
            raise ValueError(
                "Modal objective requires distinct positive forms, positive scale/time and quadrature >=64"
            )
        rule = cast(
            Callable[[int], tuple[NDArray[np.float64], NDArray[np.float64]]],
            np.polynomial.legendre.leggauss,
        )
        nodes, weights = rule(quadrature)
        x = torch.tensor((nodes + 1) / 2, dtype=torch.float64)[:, None].requires_grad_()
        integration = torch.tensor(weights / 2, dtype=torch.float64)[:, None]
        projections, metric_weights, floors = [], [], []
        for shape in forms:
            b, a = torch.full_like(x, shape), torch.full_like(x, scale)
            length = torch.log1p(torch.tensor(1 / shape, dtype=torch.float64))
            r = 1 + x / b
            n = torch.arange(2, 33, dtype=torch.float64)[None, :]
            phi = r**-0.5 * torch.sin(torch.pi * n * torch.log1p(x / b) / length)
            f = initial([x, a, b], torch.zeros_like(x))["u"]
            fx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
            fxx = torch.autograd.grad(fx.sum(), x, create_graph=True)[0]
            lf = a * r**2 * fxx + 2 * a / b * r * fx
            eigen = scale / shape**2 * ((torch.pi * n / length) ** 2 + 0.25)
            first = scale / shape**2 * ((torch.pi / length) ** 2 + 0.25)
            coefficients = 2 / (shape * length) * (integration * f * phi).sum(dim=0)
            temporal = -torch.expm1(-2 * first * time_end) / (2 * first)
            weights_n = shape * length / 2 * temporal * (eigen - first).square().squeeze(0)
            # Проецируем остаток непосредственно, без вычитания двух близких энергий.
            initial_residual = -(lf + first * f) - (coefficients * (eigen - first) * phi).sum(
                dim=1, keepdim=True
            )
            floor = temporal * (integration * initial_residual.square()).sum()
            projections.append(coefficients.detach())
            metric_weights.append(weights_n.detach())
            floors.append(floor.detach())
        self.register_buffer("forms", torch.tensor(forms, dtype=torch.float64)[:, None])
        self.register_buffer("targets", torch.stack(projections))
        self.register_buffer("metric_weights", torch.stack(metric_weights))
        self.register_buffer("truncation_floor", torch.stack(floors))

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.shape != self.targets.shape:
            raise ValueError("Modal coefficient matrix must have shape (forms,31)")
        return cast(
            torch.Tensor,
            ((coefficients - self.targets).square() * self.metric_weights).sum(dim=1).mean(),
        )

    def full_integral(self, coefficients: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self(coefficients) + self.truncation_floor.mean())
