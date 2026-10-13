"""Heat equation on the unit cube, with homogeneous Dirichlet boundaries."""

import math
from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def initial(c: Coordinates) -> TensorFields:
    return {
        "u": torch.sin(c.x.mul(math.pi))
        * torch.sin(c.y.mul(math.pi))
        * torch.sin(c.z.mul(math.pi))
    }


def heat(e: Equation) -> torch.Tensor:
    return cast(torch.Tensor, e.dt("u") - e.laplacian("u") * 0.1)


problem = Problem(fields=("u",), equations={"heat": heat}, initial=initial)
