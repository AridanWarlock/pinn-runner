"""Заданные IC и BC смешанного опыта; полного решения во входе обучения нет."""

from typing import cast

import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields


def initial(c: Coordinates) -> TensorFields:
    return {"u": torch.cos(c.x.mul(torch.pi)) * torch.cos(c.y.mul(torch.pi))}


def right(c: Coordinates) -> torch.Tensor:
    return cast(
        torch.Tensor, -torch.exp(c.t.mul(-0.2 * torch.pi**2)) * torch.cos(c.y.mul(torch.pi))
    )


def heat(e: Equation) -> torch.Tensor:
    return cast(torch.Tensor, e.dt("u") - 0.1 * e.laplacian("u"))


problem = Problem(
    fields=("u",), initial=initial, equations={"heat": heat}, boundary_values={"right": right}
)
