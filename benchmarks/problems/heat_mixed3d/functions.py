"""Cartesian heat with changing compatible operator data; no analytical teacher."""

import math
from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields

ALPHA = {"x": 0.0, "y": math.pi / 2, "z": 0.0}
FREQUENCY = {"x": math.pi, "y": math.pi / 2, "z": math.pi}
SHIFT = {"x": 0.0, "y": 0.5, "z": 0.0}


def initial(c: Coordinates) -> TensorFields:
    mode = torch.ones_like(c.t)
    for axis, column in c.spatial.items():
        mode = mode * column.sub(SHIFT[axis]).mul(FREQUENCY[axis]).cos()
    return {"u": mode.add(1)}


def data(c: Coordinates, axis: str) -> torch.Tensor:
    background = torch.ones_like(c.t)
    for column in c.spatial.values():
        background = background + c.t * column
    return cast(torch.Tensor, ALPHA[axis] * background + c.spatial[axis].mul(2).sub(1).mul(c.t))


def wall_x(c: Coordinates) -> torch.Tensor:
    return data(c, "x")


def wall_y(c: Coordinates) -> torch.Tensor:
    return data(c, "y")


def wall_z(c: Coordinates) -> torch.Tensor:
    return data(c, "z")


def heat(e: Equation) -> torch.Tensor:
    source = torch.zeros_like(e.coordinates.t)
    for column in e.coordinates.spatial.values():
        source = source + column
    return cast(torch.Tensor, e.dt("u") - 0.1 * e.laplacian("u") - source)


problem = Problem(
    fields=("u",),
    equations={"heat": heat},
    initial=initial,
    boundary_values={"wall_x": wall_x, "wall_y": wall_y, "wall_z": wall_z},
)
