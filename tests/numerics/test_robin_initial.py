"""Physical initial and changing Robin traces survive the model transformation."""

import math
import os
from pathlib import Path

import pytest
import torch

from benchmarks.cases import CASES
from pinnstorch.api import Coordinates, Equation
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings


@pytest.mark.parametrize("name", ["heat_neumann", "heat_robin"])
def test_factory_model_exact_initial_and_changing_boundary(name: str, tmp_path: Path) -> None:
    case = CASES[name]
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    config = case.config(device, 3, 19)
    config["training"]["constraints"] = "hard_initial_robin"
    (tmp_path / "functions.py").write_bytes(case.source.read_bytes())
    problem = load_problem(tmp_path / "functions.py")
    factory = SolverFactory(TaskSettings.read(config), problem, tmp_path)
    model = factory.build(True).model.to(device)
    x = torch.tensor([[0.0], [0.2], [1.0]], device=device, requires_grad=True)
    time = torch.full_like(x, 0.3, requires_grad=True)
    physical = model([x], time)
    equation = Equation(physical, Coordinates({"x": x}, time))
    derivative = equation.diff("u", "x")
    for index, normal in ((0, -1), (-1, 1)):
        value = derivative[index] * normal
        if name == "heat_robin":
            value = value + math.pi / 2 * physical["u"][index]
        target = problem.boundary_values["wall"](Coordinates({"x": x}, time))[index]
        torch.testing.assert_close(value, target, rtol=2e-5, atol=2e-5)
    initial_time = torch.zeros_like(time)
    torch.testing.assert_close(
        model([x], initial_time)["u"],
        problem.initial(Coordinates({"x": x}, initial_time))["u"],
        rtol=0,
        atol=0,
    )
    loss = factory.adapter.residuals(physical, x, time)["heat"].square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        prediction = model([x.detach()], time.detach())["u"]
        assert not prediction.requires_grad
        torch.testing.assert_close(prediction, physical["u"], rtol=0, atol=0)
    assert factory.constraints.remaining_losses == ("equations",)


@pytest.mark.parametrize(
    "invalid", ["initial", "missing", "duplicate", "beta", "dimensions", "execution"]
)
def test_invalid_robin_configuration_rejected(tmp_path: Path, invalid: str) -> None:
    case = CASES["heat_robin"]
    config = case.config("cpu", 3, 19)
    config["training"]["constraints"] = "hard_initial_robin"
    source = case.source.read_text()
    if invalid == "initial":
        source = source.replace(".cos().add(1)", ".cos().add(2)")
    elif invalid == "missing":
        config["boundaries"][0]["side"] = "lower"
    elif invalid == "duplicate":
        config["boundaries"].append(dict(config["boundaries"][0]))
    elif invalid == "beta":
        config["boundaries"][0]["beta"] = 0
    elif invalid == "dimensions":
        config["domain"]["space"]["y"] = dict(config["domain"]["space"]["x"])
    else:
        config["training"]["hard_execution"] = "product_rule"
    (tmp_path / "functions.py").write_text(source)
    with pytest.raises(ValueError):
        SolverFactory(TaskSettings.read(config), load_problem(tmp_path / "functions.py"), tmp_path)
