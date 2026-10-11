"""1D heat equation for a short CPU runner test."""

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def heat(eq: Equation) -> torch.Tensor:
    return eq.dt("T") - 0.3 * eq.laplacian("T")


def initial(c: Coordinates) -> TensorFields:
    return {"T": torch.sin(torch.pi * c.x)}


problem = Problem(fields=("T",), equations={"heat": heat}, initial=initial)
