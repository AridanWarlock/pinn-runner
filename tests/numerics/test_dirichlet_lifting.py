"""Shifted box lifting, exact traces, physical derivatives and training gradients."""

import os
from dataclasses import replace
from functools import partial
from itertools import product
from pathlib import Path

import pytest
import torch

from benchmarks.cases import CASES
from pinnstorch.api import Coordinates, Equation
from pinnstorch.contracts import TensorOutputs
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings
from pinnstorch.training.constraints import HardBoxInitial
from pinnstorch.training.dirichlet_lifting import DirichletLifting, FaceTrace
from pinnstorch.training.hard_execution import ProductRuleHard


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_lifting_and_product_rule(dimensions: int, dtype: torch.dtype) -> None:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    bounds = tuple((-2.0 + i, 1.0 + 2 * i) for i in range(dimensions))
    names = ("x", "y", "z")[:dimensions]
    weight = torch.tensor(0.7, device=device, dtype=dtype, requires_grad=True)

    def reference(spatial: list[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        return weight * time.square() + sum(column.square() for column in spatial)

    def face(
        axis: int, side: int, spatial: list[torch.Tensor], time: torch.Tensor
    ) -> torch.Tensor:
        projected = list(spatial)
        projected[axis] = torch.full_like(time, bounds[axis][side])
        return reference(projected, time)

    traces = tuple(
        FaceTrace("u", axis, side, partial(face, axis, side))
        for axis in range(dimensions)
        for side in (0, 1)
    )
    lifting = DirichletLifting.build(bounds, ("u",), traces)

    def initial(spatial: list[torch.Tensor], time: torch.Tensor) -> TensorOutputs:
        return {"u": reference(spatial, torch.full_like(time, 2.0))}

    recipe = HardBoxInitial(bounds, (2, 5), initial, spatial_names=names, lifting=lifting)
    spatial_points = list(
        product(*[(left, left + 1e-5, (left + right) / 2, right) for left, right in bounds])
    )
    columns = [
        torch.tensor(
            [[row[i]] for row in spatial_points], device=device, dtype=dtype, requires_grad=True
        )
        for i in range(dimensions)
    ]
    time = torch.full_like(columns[0], 3.2, requires_grad=True)
    fields = {"u": torch.sin(sum(columns) * weight + time)}
    coordinates = Coordinates(dict(zip(names, columns)), time)
    baseline = replace(recipe, lifting=lifting.__call__)
    eager = Equation(baseline(fields, *columns, time), coordinates)
    fused = Equation(recipe(fields, *columns, time), coordinates)
    optimized = Equation(ProductRuleHard(recipe)(fields, *columns, time), coordinates)
    tolerance = 2e-11 if dtype == torch.float64 else 3e-4
    torch.testing.assert_close(fused.field("u"), eager.field("u"), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        optimized.field("u"), eager.field("u"), rtol=tolerance, atol=tolerance
    )
    for axis in (*names, "t"):
        for order in (1, 2):
            actual, expected = optimized.diff("u", axis, order), eager.diff("u", axis, order)
            torch.testing.assert_close(
                fused.diff("u", axis, order), expected, rtol=tolerance, atol=tolerance
            )
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(
                torch.autograd.grad(actual.square().mean(), weight, retain_graph=True)[0],
                torch.autograd.grad(expected.square().mean(), weight, retain_graph=True)[0],
                rtol=tolerance,
                atol=tolerance,
            )
    for axis in range(dimensions):
        for side in (0, 1):
            projected = lifting.project(columns, axis, side)
            output = recipe(fields, *projected, time)["u"]
            torch.testing.assert_close(
                output, reference(projected, time), rtol=tolerance, atol=tolerance
            )
    initial_time = torch.full_like(time, 2.0)
    torch.testing.assert_close(
        recipe(fields, *columns, initial_time)["u"],
        initial(columns, initial_time)["u"],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("invalid", ["initial", "corner", "duplicate"])
def test_incompatible_traces_rejected(tmp_path: Path, invalid: str) -> None:
    case = CASES["heat2d"]
    config = case.config("cpu", 3, 19)
    config["training"]["constraints"] = "hard_initial_dirichlet"
    config["boundaries"] = [
        {"kind": "dirichlet", "field": "u", "axis": axis, "value": 1.0} for axis in ("x", "y")
    ]
    source = case.source.read_text().replace('return {"u":', 'return {"u": 1 +')
    if invalid == "initial":
        config["boundaries"][0]["value"] = 2.0
    elif invalid == "corner":
        source += '\ndef wall(c: Coordinates) -> torch.Tensor:\n    return 1 + c.t\nproblem.boundary_values["wall"] = wall\n'
        config["boundaries"][0].pop("value")
        config["boundaries"][0]["value_function"] = "wall"
    else:
        config["boundaries"].append(dict(config["boundaries"][0]))
    (tmp_path / "functions.py").write_text(source)
    with pytest.raises(ValueError, match="Incompatible|one condition"):
        SolverFactory(TaskSettings.read(config), load_problem(tmp_path / "functions.py"), tmp_path)


def test_nonzero_boundary_values_change_checkpoint_identity(tmp_path: Path) -> None:
    case = CASES["heat_dirichlet2d"]
    source = (
        case.source.read_text()
        + '\ndef warmer(c: Coordinates) -> torch.Tensor:\n    return wall(c) + c.t\nproblem.boundary_values["warmer"] = warmer\n'
    )
    (tmp_path / "functions.py").write_text(source)
    config = case.config("cpu", 3, 19)
    config["training"]["constraints"] = "hard_initial_dirichlet"
    settings = TaskSettings.read(config)
    factory = SolverFactory(settings, load_problem(tmp_path / "functions.py"), tmp_path)
    assert factory.constraints.recipe is not None
    assert factory.constraints.recipe.lifting is not None
    assert all(face.value_function == "wall" for face in factory.constraints.exact_faces)
    signature = factory.constraint_signature
    # Reordering equivalent face declarations does not change identity.
    config["boundaries"].reverse()
    other = SolverFactory(TaskSettings.read(config), factory.problem, tmp_path)
    assert other.constraint_signature == signature

    # IC remains compatible at t0, but different later wall temperatures
    # define a different physical checkpoint even with identical source bytes.
    for face in config["boundaries"]:
        face["value_function"] = "warmer"
    changed = SolverFactory(TaskSettings.read(config), factory.problem, tmp_path)
    assert changed.constraint_signature != signature
