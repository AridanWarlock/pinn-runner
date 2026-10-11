"""Heat equation family on a square, conditioned on conductivity."""

import numpy as np
import torch
from numpy.typing import NDArray

from pinnstorch.utils import gradient


def initial_conditions(
    spatial: NDArray[np.float64], *, parameters: dict[str, NDArray[np.float64]]
) -> dict[str, NDArray[np.float64]]:
    return {"T": np.prod(np.sin(np.pi * spatial), axis=1, keepdims=True)}


def pde_fn(
    outputs: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    *,
    parameters: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    dx, dy, dt = gradient(outputs["T"], [x, y, t])
    outputs["residual"] = dt - parameters["conductivity"] * (
        gradient(dx, x)[0] + gradient(dy, y)[0]
    )
    return outputs
