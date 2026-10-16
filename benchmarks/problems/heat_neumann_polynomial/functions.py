"""Polynomial initial data and forcing; no analytical time-dependent teacher."""

from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def profile(c: Coordinates) -> torch.Tensor:
    value = torch.ones_like(c.t)
    for column in c.spatial.values():
        value = value * (3 * column.square() - 2 * column.pow(3))
    return value


def initial(c: Coordinates) -> TensorFields:
    return {"u": profile(c)}


def source(c: Coordinates) -> torch.Tensor:
    # Explicit polynomial Laplacian, independent of network derivatives.
    columns = tuple(c.spatial.values())
    laplacian = torch.zeros_like(c.t)
    for axis, column in enumerate(columns):
        term = column.mul(-12).add(6)
        for other, coordinate in enumerate(columns):
            if other != axis:
                term = term * (3 * coordinate.square() - 2 * coordinate.pow(3))
        laplacian = laplacian + term
    return cast(torch.Tensor, c.t.neg().exp() * (-profile(c) - 0.1 * laplacian))


def heat(e: Equation) -> torch.Tensor:
    return cast(torch.Tensor, e.dt("u") - 0.1 * e.laplacian("u") - source(e.coordinates))


problem = Problem(fields=("u",), equations={"heat": heat}, initial=initial)
