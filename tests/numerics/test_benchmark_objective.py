"""A nonzero exact field has zero PDE objective: no hidden solution-value penalty."""

from pathlib import Path

import torch

from benchmarks.cases import CASES
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings
from scripts.benchmark_robin_execution import objective
from scripts.benchmark_robin_lifting import signature


def test_benchmark_uses_only_equation_residuals(tmp_path: Path) -> None:
    source = """import torch
from pinnstorch.api import Coordinates, Equation, Problem, TensorFields

def initial(c: Coordinates) -> TensorFields:
    return {"u": torch.ones_like(c.t)}

def heat(e: Equation) -> torch.Tensor:
    return e.dt("u") - 0.1 * e.laplacian("u") - torch.ones_like(e.coordinates.t)

problem = Problem(fields=("u",), equations={"heat": heat}, initial=initial)
"""
    path = tmp_path / "functions.py"
    path.write_text(source)
    config = CASES["heat_neumann"].config("cpu", 1, 19)
    config["training"].update(constraints="hard_initial_robin", hard_lifting="expanded")
    config["network"]["hidden_layers"] = [8, 8]
    config["boundaries"] = [
        {"kind": "neumann", "field": "u", "axis": "x", "side": "both", "value": 0}
    ]
    factory = SolverFactory(TaskSettings.read(config), load_problem(path), tmp_path)
    model = factory.build(True).model
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        final_layer = [
            layer for layer in model.net.modules() if isinstance(layer, torch.nn.Linear)
        ][-1]
        assert final_layer.bias is not None
        final_layer.bias.fill_(0.5)
    points = torch.tensor([[0.2, 0.1], [0.7, 0.4]])
    loss = objective(factory, model, points)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=0, rtol=0)
    measured = signature(factory, model, points)
    assert bool((measured["field"] > 1).all())
    assert measured["residual"].shape == (2, 1)
    torch.testing.assert_close(measured["residual"], torch.zeros((2, 1)), atol=0, rtol=0)
    torch.testing.assert_close(
        measured["parameter_gradient"],
        torch.zeros_like(measured["parameter_gradient"]),
        atol=0,
        rtol=0,
    )
