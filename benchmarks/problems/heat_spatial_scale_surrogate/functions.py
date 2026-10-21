"""Unforced spatial family: k(x)=a[1+2x(1-x)], with a as the shared input."""

from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def initial(c: Coordinates) -> TensorFields:
    return {"u": 4 * c.x * (1 - c.x)}


def conductivity(c: Coordinates) -> torch.Tensor:
    return cast(torch.Tensor, c.parameters["a"] * (1 + 2 * c.x * (1 - c.x)))


def transform(e: Equation) -> TensorFields:
    x, t = e.coordinates.x, e.coordinates.t
    return {"u": 4 * x * (1 - x) + t * x * (1 - x) * e.field("raw")}


def heat(e: Equation) -> torch.Tensor:
    return cast(
        torch.Tensor,
        e.dt("u") - e.divergence({"x": e.coefficients["conductivity"] * e.diff("u", "x")}),
    )


problem = Problem(
    fields=("u",),
    network_fields=("raw",),
    initial=initial,
    coefficients={"conductivity": conductivity},
    equations={"heat": heat},
    transform=transform,
)
