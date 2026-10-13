"""Nonzero time-dependent walls: heat without forcing or a solution teacher."""

import math
from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def wall(c: Coordinates) -> torch.Tensor:
    value: torch.Tensor = torch.ones_like(c.t).add(c.t.mul(0.2 * len(c.spatial)))
    for column in c.spatial.values():
        value = value.add(column.square())
    return value


def initial(c: Coordinates) -> TensorFields:
    mode = torch.ones_like(c.t)
    for column in c.spatial.values():
        mode = mode.mul(column.mul(math.pi).sin())
    return {"u": wall(c).add(mode)}


def heat(e: Equation) -> torch.Tensor:
    return cast(torch.Tensor, e.dt("u") - 0.1 * e.laplacian("u"))


problem = Problem(
    fields=("u",), equations={"heat": heat}, initial=initial, boundary_values={"wall": wall}
)
