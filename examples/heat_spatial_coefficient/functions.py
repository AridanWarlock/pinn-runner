"""Heat diffusion with a prescribed spatial conductivity tensor."""

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def heat(e: Equation) -> torch.Tensor:
    conductivity = e.coefficients["conductivity"]
    flux = {axis: conductivity * derivative for axis, derivative in e.gradient("T").items()}
    return e.dt("T") - e.divergence(flux)


def initial(c: Coordinates) -> TensorFields:
    return {"T": torch.sin(torch.pi * c.x) * torch.sin(torch.pi * c.y)}


problem = Problem(fields=("T",), equations={"heat": heat}, initial=initial)
