"""Shared diagnostics must use the actual source of each declared analytical case."""

import numpy as np
import pytest
import torch

from benchmarks.cases import CASES, BoxNeumannHeatCase
from benchmarks.modal_balance import AffineHeatSolution, HeatEigenmode
from pinnstorch.api import Coordinates, Equation
from pinnstorch.runner.problem import load_problem


@pytest.mark.parametrize(
    "name",
    [
        "heat_neumann2d",
        "heat_neumann3d",
        "heat_robin2d",
        "heat_robin3d",
        "heat_neumann_homogeneous1d",
        "heat_neumann_homogeneous2d",
        "heat_neumann_homogeneous3d",
    ],
)
def test_tensor_reference_matches_numpy_and_declared_problem(name: str) -> None:
    case = CASES[name]
    dimensions = case.dimensions
    assert isinstance(case, BoxNeumannHeatCase)
    names = tuple("xyz"[:dimensions])
    modes = tuple(case.mode_for(name) for name in names)
    mode = HeatEigenmode(tuple(m.frequency for m in modes), tuple(m.shift for m in modes))
    analytical = AffineHeatSolution(mode, case.affine_background_rate)
    columns = [
        torch.linspace(0.13 + i * 0.02, 0.87, 7, dtype=torch.float64)
        .reshape(-1, 1)
        .requires_grad_()
        for i in range(dimensions)
    ]
    clock = torch.linspace(0.03, 0.47, 7, dtype=torch.float64).reshape(-1, 1).requires_grad_()
    reference = analytical.reference(columns, clock)
    numpy_reference = case.reference(
        {name: x.detach().numpy() for name, x in zip(names, columns)}, clock.detach().numpy()
    )
    np.testing.assert_allclose(reference.detach().numpy(), numpy_reference, rtol=1e-12, atol=1e-12)
    coordinates = Coordinates(dict(zip(names, columns)), clock)
    problem = load_problem(case.source)
    residual = problem.equations["heat"](Equation({"u": reference}, coordinates))
    torch.testing.assert_close(residual, torch.zeros_like(clock), rtol=0, atol=2e-12)
    if name.startswith("heat_neumann_homogeneous"):
        assert torch.equal(analytical.source(columns, clock), torch.zeros_like(clock))
