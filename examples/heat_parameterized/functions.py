"""2D heat equation using the public problem language."""

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def heat(eq: Equation) -> torch.Tensor:
    return eq.dt("T") - eq.parameters["conductivity"] * eq.laplacian("T")


def initial(c: Coordinates) -> TensorFields:
    return {"T": torch.sin(torch.pi * c.x) * torch.sin(torch.pi * c.y)}


problem = Problem(fields=("T",), equations={"heat": heat}, initial=initial)
