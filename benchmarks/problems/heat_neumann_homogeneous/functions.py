"""Unforced heat with homogeneous Neumann faces; independent of dimension."""

import math
from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def initial(c: Coordinates) -> TensorFields:
    mode = torch.ones_like(c.t)
    for coordinate in c.spatial.values():
        mode = mode * coordinate.mul(math.pi).cos()
    return {"u": mode.add(1)}


def heat(e: Equation) -> torch.Tensor:
    return cast(torch.Tensor, e.dt("u").sub(e.laplacian("u").mul(0.1)))


problem = Problem(fields=("u",), equations={"heat": heat}, initial=initial)
