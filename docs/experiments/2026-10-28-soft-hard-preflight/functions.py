"""Заданные физические данные; полного решения во входе обучения нет."""

import torch

from pinnstorch.api import Coordinates, Equation, Problem


def initial(c: Coordinates):
    return {"u": torch.cos(torch.pi * c.x) * torch.cos(torch.pi * c.y)}


def right(c: Coordinates):
    return -torch.exp(-0.2 * torch.pi**2 * c.t) * torch.cos(torch.pi * c.y)


def heat(e: Equation):
    return e.dt("u") - 0.1 * e.laplacian("u")


problem = Problem(
    fields=("u",), initial=initial, equations={"heat": heat}, boundary_values={"right": right}
)
