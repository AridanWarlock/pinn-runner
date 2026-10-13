"""Right-going transport, with a prescribed inflow and no outflow constraint."""

import math
from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def initial(c: Coordinates) -> TensorFields:
    return {"u": torch.sin(c.x.mul(math.pi))}


def inflow(c: Coordinates) -> torch.Tensor:
    return -torch.sin(c.t.mul(math.pi))


def transport(e: Equation) -> torch.Tensor:
    return cast(torch.Tensor, e.dt("u") + e.diff("u", "x"))


problem = Problem(
    fields=("u",),
    equations={"transport": transport},
    initial=initial,
    boundary_values={"inflow": inflow},
)
