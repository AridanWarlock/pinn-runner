"""Viscous Burgers equation with a manufactured, prescribed source term."""

import math
from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def initial(c: Coordinates) -> TensorFields:
    return {"u": torch.sin(c.x.mul(math.pi))}


def burgers(e: Equation) -> torch.Tensor:
    c = e.coordinates
    known = torch.exp(-c.t) * torch.sin(c.x.mul(math.pi))
    known_dx = torch.exp(-c.t) * math.pi * torch.cos(c.x.mul(math.pi))
    source = known * (-1 + 0.1 * math.pi**2) + known * known_dx
    return cast(
        torch.Tensor, e.dt("u") + e.field("u") * e.diff("u", "x") - e.laplacian("u") * 0.1 - source
    )


problem = Problem(fields=("u",), equations={"burgers": burgers}, initial=initial)
