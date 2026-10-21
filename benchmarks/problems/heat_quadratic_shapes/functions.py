"""Теплопроводность с независимым изменением масштаба и формы k(x)."""

from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def conductivity(c: Coordinates) -> torch.Tensor:
    return cast(torch.Tensor, c.parameters["a"] * (1 + c.x / c.parameters["b"]) ** 2)


def initial(c: Coordinates) -> TensorFields:
    return {"u": torch.sin(c.x.mul(torch.pi)) ** 3}


def transform(e: Equation) -> TensorFields:
    x, t = e.coordinates.x, e.coordinates.t
    return {"u": torch.sin(x.mul(torch.pi)) ** 3 + t * x * (1 - x) * e.field("raw")}


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
