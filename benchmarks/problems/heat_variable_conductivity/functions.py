"""Prescribed affine conductivity and polynomial forcing, without teacher data."""

from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields
from pinnstorch.coefficients import RegularGridCoefficient

DIMENSIONS = 2
BETA = 1
REPRESENTATION = "formula"
K0 = 1 / 8


def profile(c: Coordinates) -> torch.Tensor:
    if tuple(c.spatial) != tuple("xy"[:DIMENSIONS]):
        raise ValueError("This benchmark requires its declared unit-box spatial axes")
    result = torch.ones_like(c.t)
    for x in c.spatial.values():
        result = result * x.square() * (3 - 2 * x)
    return result


def initial(c: Coordinates) -> TensorFields:
    return {"u": profile(c)}


def conductivity(c: Coordinates) -> torch.Tensor:
    result = torch.ones_like(c.t)
    for x in c.spatial.values():
        result = result.add(x.mul(BETA / DIMENSIONS))
    return cast(torch.Tensor, result.mul(K0))


def source(c: Coordinates) -> torch.Tensor:
    # Closed polynomial derivatives of the prescribed profile, not of the network.
    xs = tuple(c.spatial.values())
    laplacian = torch.zeros_like(c.t)
    gradient_sum = torch.zeros_like(c.t)
    for i, x in enumerate(xs):
        others = torch.ones_like(c.t)
        for j, y in enumerate(xs):
            if i != j:
                others = others * y.square() * (3 - 2 * y)
        laplacian = laplacian.add(x.mul(-12).add(6).mul(others))
        gradient_sum = gradient_sum + 6 * x * (1 - x) * others
    return cast(
        torch.Tensor,
        (-c.t).exp()
        * (-profile(c) - conductivity(c) * laplacian - gradient_sum.mul(K0 * BETA / DIMENSIONS)),
    )


def heat(e: Equation) -> torch.Tensor:
    flux = {
        axis: e.coefficients["conductivity"] * derivative
        for axis, derivative in e.gradient("u").items()
    }
    return cast(torch.Tensor, e.dt("u") - e.divergence(flux) - source(e.coordinates))


def table() -> RegularGridCoefficient:
    axes = {"x": torch.tensor([0.0, 0.125, 0.5, 1.0])}
    if DIMENSIONS == 2:
        axes["y"] = torch.tensor([0.0, 0.25, 0.75, 1.0])
    grids = torch.meshgrid(*axes.values(), indexing="ij")
    values = torch.ones_like(grids[0])
    for grid in grids:
        values = values.add(grid.mul(BETA / DIMENSIONS))
    return RegularGridCoefficient(axes, values.mul(K0))


if DIMENSIONS not in (1, 2) or BETA not in (0, 1):
    raise ValueError("Only dimensions1/2 and beta0/1 are registered for this benchmark")
if REPRESENTATION not in ("formula", "table"):
    raise ValueError("Unknown conductivity representation")

problem = Problem(
    fields=("u",),
    equations={"heat": heat},
    initial=initial,
    coefficients={"conductivity": conductivity if REPRESENTATION == "formula" else table()},
)
