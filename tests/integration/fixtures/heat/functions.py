"""Minimal user-supplied heat equation for the runner integration test."""

import numpy as np
import torch
from numpy.typing import NDArray

from pinnstorch.utils import gradient


def initial_conditions(spatial: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    return {"T": np.sin(np.pi * spatial[:, 0:1])}


def pde_fn(
    outputs: dict[str, torch.Tensor], x: torch.Tensor, t: torch.Tensor
) -> dict[str, torch.Tensor]:
    dx, dt = gradient(outputs["T"], [x, t])
    outputs["residual"] = dt - 0.3 * gradient(dx, x)[0]
    return outputs


def boundary(
    outputs: dict[str, torch.Tensor], x: list[torch.Tensor], t: torch.Tensor
) -> dict[str, torch.Tensor]:
    outputs["boundary"] = outputs["T"]
    return outputs


boundary_functions = {"boundary": boundary}
