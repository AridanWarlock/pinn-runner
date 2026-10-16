"""Explicit Neumann coordinate features; physical derivatives follow the full graph."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from pinnstorch.contracts import TensorOutputs

from .neural_net import FCN


@dataclass(frozen=True)
class NeumannFeatures:
    bounds: tuple[tuple[float, float], ...]
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in ("neumann_cosine", "neumann_polynomial"):
            raise ValueError("Unknown Neumann spatial encoding")
        if not self.bounds or any(
            not (math.isfinite(lower) and math.isfinite(upper) and lower < upper)
            for lower, upper in self.bounds
        ):
            raise ValueError("Spatial encoding requires increasing physical intervals")

    @property
    def width(self) -> int:
        return len(self.bounds) * (2 if self.kind == "neumann_polynomial" else 1)

    @property
    def feature_interval(self) -> tuple[float, float]:
        return (-1.0, 1.0) if self.kind == "neumann_cosine" else (0.0, 1.0)

    def __call__(self, spatial: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(spatial) != len(self.bounds):
            raise ValueError("Spatial encoding coordinate count differs from its bounds")
        result: list[torch.Tensor] = []
        for column, (lower, upper) in zip(spatial, self.bounds):
            x = column.sub(lower).div(upper - lower)
            if self.kind == "neumann_cosine":
                result.append(x.mul(math.pi).cos())
            else:
                result.extend(
                    [x.square().mul(3 - 2 * x), x.square().mul((1 - x).square()).mul(16)]
                )
        return result


class FeatureFCN(nn.Module):
    def __init__(
        self,
        layers: Sequence[int],
        bounds: tuple[tuple[float, float], ...],
        time_interval: tuple[float, float],
        output_names: Sequence[str],
        encoding: str,
    ) -> None:
        super().__init__()
        self.features = NeumannFeatures(bounds, encoding)
        lower, upper = self.features.feature_interval
        width = self.features.width
        self.network = FCN(
            [width + 1, *layers[1:]],
            [lower] * width + [time_interval[0]],
            [upper] * width + [time_interval[1]],
            output_names,
        )

    def forward(self, spatial: list[torch.Tensor], time: torch.Tensor) -> TensorOutputs:
        return self.network(self.features(spatial), time)
