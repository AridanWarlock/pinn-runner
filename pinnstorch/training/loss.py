"""Named objective composition and read-only gradient diagnostics."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch


class LossPolicy(Protocol):
    def combine(self, terms: Mapping[str, torch.Tensor]) -> torch.Tensor: ...


@dataclass(frozen=True)
class EqualWeights:
    def combine(self, terms: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return sum_terms(terms)


@dataclass(frozen=True)
class FixedWeights:
    weights: Mapping[str, float]

    def combine(self, terms: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return sum_terms(
            {name: value.mul(self.weights.get(name, 1.0)) for name, value in terms.items()}
        )


def sum_terms(terms: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if not terms:
        raise ValueError("Training requires at least one loss term")
    values = iter(terms.values())
    total = next(values)
    for value in values:
        total = total + value
    return total


def gradient_norms(
    terms: Mapping[str, torch.Tensor], parameters: Sequence[torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Unweighted norms; autograd.grad leaves parameter.grad buffers untouched."""
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    results: dict[str, torch.Tensor] = {}
    for name, loss in terms.items():
        if not trainable or not loss.requires_grad:
            results[name] = loss.detach().new_zeros(())
            continue
        gradients = torch.autograd.grad(loss, trainable, retain_graph=True, allow_unused=True)
        squares = [
            gradient.detach().square().sum() for gradient in gradients if gradient is not None
        ]
        results[name] = (
            torch.stack(squares).sum().sqrt() if squares else loss.detach().new_zeros(())
        )
    return results
