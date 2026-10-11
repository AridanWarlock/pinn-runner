"""Minimal user-supplied heat equation for the runner integration test."""

from typing import Any

import torch

from pinnstorch.utils import gradient


def read_data_fn(data: dict[str, Any]) -> dict[str, Any]:
    return {"spatial": [data["x"]], "time": data["t"], "solution": {"T": data["T"]}}


def pde_fn(
    outputs: dict[str, torch.Tensor], x: torch.Tensor, t: torch.Tensor
) -> dict[str, torch.Tensor]:
    dx, dt = gradient(outputs["T"], [x, t])
    outputs["residual"] = dt - 0.3 * gradient(dx, x)[0]
    return outputs
