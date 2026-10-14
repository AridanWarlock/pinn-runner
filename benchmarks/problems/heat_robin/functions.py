"""Heat with genuine homogeneous Robin mode, changing data and source x."""

import math
from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields

MU = math.pi / 2


def initial(c: Coordinates) -> TensorFields:
    return {"u": c.x.sub(0.5).mul(MU).cos().add(1)}


def wall(c: Coordinates) -> torch.Tensor:
    return cast(torch.Tensor, c.x.mul(c.t).add(1).mul(MU).add(c.x.mul(2).sub(1).mul(c.t)))


def heat(e: Equation) -> torch.Tensor:
    return cast(torch.Tensor, e.dt("u") - 0.1 * e.laplacian("u") - e.coordinates.x)


problem = Problem(
    fields=("u",), equations={"heat": heat}, initial=initial, boundary_values={"wall": wall}
)
