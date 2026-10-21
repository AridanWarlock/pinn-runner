"""Public Problem for an unforced, two-parameter conductivity family."""

from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def conductivity(c: Coordinates) -> torch.Tensor:
    return cast(
        torch.Tensor,
        (
            torch.exp(
                c.parameters["a"] * torch.cos(c.x.mul(torch.pi))
                + c.parameters["b"] * torch.cos(c.x.mul(2 * torch.pi))
            )
            / 8
        ),
    )


def initial(c: Coordinates) -> TensorFields:
    return {"u": torch.sin(c.x.mul(torch.pi))}


def transform(e: Equation) -> TensorFields:
    x, t = e.coordinates.x, e.coordinates.t
    return {"u": torch.sin(x.mul(torch.pi)) + t * x * (1 - x) * e.field("raw")}


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
