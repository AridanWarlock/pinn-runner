"""Heat equation on the unit square; a short CPU pipeline example."""

import numpy as np
import torch
from numpy.typing import NDArray

from pinnstorch.utils import gradient


def initial_conditions(spatial: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    return {"T": np.prod(np.sin(np.pi * spatial), axis=1, keepdims=True)}


def pde_fn(
    outputs: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
) -> dict[str, torch.Tensor]:
    dx, dy, dt = gradient(outputs["T"], [x, y, t])
    outputs["residual"] = dt - 0.3 * (gradient(dx, x)[0] + gradient(dy, y)[0])
    return outputs


def boundary(
    outputs: dict[str, torch.Tensor], spatial: list[torch.Tensor], t: torch.Tensor
) -> dict[str, torch.Tensor]:
    outputs["boundary"] = outputs["T"]
    return outputs


boundary_functions = {"boundary": boundary}
