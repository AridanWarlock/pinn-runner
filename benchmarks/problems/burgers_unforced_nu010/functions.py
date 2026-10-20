"""Unforced viscous Burgers on [-1,1] x [0,1], fixed viscosity 0.1."""

import math
from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def initial(c: Coordinates) -> TensorFields:
    return {"u": -torch.sin(c.x.mul(math.pi))}


def burgers(e: Equation) -> torch.Tensor:
    return cast(torch.Tensor, e.dt("u") + e.field("u") * e.diff("u", "x") - 0.1 * e.laplacian("u"))


problem = Problem(fields=("u",), equations={"burgers": burgers}, initial=initial)
