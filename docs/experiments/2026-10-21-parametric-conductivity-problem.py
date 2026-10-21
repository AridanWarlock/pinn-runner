"""Public Problem for an unforced, two-parameter conductivity family."""

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def conductivity(c: Coordinates) -> torch.Tensor:
    return (
        torch.exp(
            c.parameters["a"] * torch.cos(torch.pi * c.x)
            + c.parameters["b"] * torch.cos(2 * torch.pi * c.x)
        )
        / 8
    )


def initial(c: Coordinates) -> TensorFields:
    return {"u": torch.sin(torch.pi * c.x)}


def transform(e: Equation) -> TensorFields:
    x, t = e.coordinates.x, e.coordinates.t
    return {"u": torch.sin(torch.pi * x) + t * x * (1 - x) * e.field("raw")}


def heat(e: Equation) -> torch.Tensor:
    return e.dt("u") - e.divergence({"x": e.coefficients["conductivity"] * e.diff("u", "x")})


problem = Problem(
    fields=("u",),
    network_fields=("raw",),
    initial=initial,
    coefficients={"conductivity": conductivity},
    equations={"heat": heat},
    transform=transform,
)
