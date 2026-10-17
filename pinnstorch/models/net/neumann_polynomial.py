"""Identical polynomial features with an optional analytic, differentiable pullback."""

from collections.abc import Callable
from typing import Protocol, cast

import torch


class FeatureContext(Protocol):
    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


def polynomial_features(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return q and the scaled feature P=16p without changing operation order."""
    return x.square().mul(3 - 2 * x), x.square().mul((1 - x).square()).mul(16)


class _AnalyticPolynomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FeatureContext, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(x)
        return polynomial_features(x)

    @staticmethod
    def backward(
        ctx: FeatureContext, grad_q: torch.Tensor, grad_scaled_p: torch.Tensor
    ) -> tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        # Save the input, not no-grad forward intermediates: higher derivatives need x.
        r = x * (1 - x)
        return (r * (6 * grad_q + 32 * (1 - 2 * x) * grad_scaled_p),)


def analytic_polynomial_features(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    apply = cast(
        Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]], _AnalyticPolynomial.apply
    )
    return apply(x)
