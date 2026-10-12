"""Explicit backend acceptance; selected by scripts/check_backend.py only."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from numpy.typing import NDArray
from scipy.io import loadmat

from pinnstorch.api import Coordinates, Equation
from pinnstorch.coefficients import RegularGridCoefficient
from tests.integration.test_runner_files import launch, prepare_input

DEVICE = os.environ.get("PINN_TEST_DEVICE", "cpu")


def evaluate(device: str, dimensions: int) -> tuple[torch.Tensor, torch.Tensor]:
    spatial = {
        name: torch.linspace(0.13, 0.87, 11, device=device).reshape(-1, 1).requires_grad_()
        for name in ("x", "y", "z")[:dimensions]
    }
    time = torch.full_like(spatial["x"], 0.2, requires_grad=True)
    coordinates = Coordinates(spatial, time)
    axes = {name: torch.tensor([0.0, 0.4, 1.0]) for name in spatial}
    nodes = torch.meshgrid(*axes.values(), indexing="ij")
    table = RegularGridCoefficient(axes, 1 + torch.stack(nodes).sum(dim=0)).to(device)
    coefficient = table(coordinates)
    amplitude = torch.tensor(0.7, device=device, requires_grad=True)
    squared_radius = sum(value.square() for value in spatial.values())
    field = amplitude * (squared_radius + time)
    equation = Equation({"T": field}, coordinates)
    residual = equation.dt("T") - equation.divergence(
        {axis: coefficient * value for axis, value in equation.gradient("T").items()}
    )
    expected = amplitude * (1 - 2 * dimensions * coefficient - 2 * sum(spatial.values()))
    torch.testing.assert_close(residual, expected, rtol=2e-5, atol=2e-5)
    derivative = torch.autograd.grad(residual.square().mean(), amplitude)[0]
    assert torch.isfinite(derivative)
    return residual.detach().cpu(), derivative.detach().cpu()


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_conservative_operator_and_backward(dimensions: int) -> None:
    expected = evaluate("cpu", dimensions)
    actual = evaluate(DEVICE, dimensions)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("training_device", list(dict.fromkeys(("cpu", DEVICE))))
def test_train_and_checkpoint_transfer(tmp_path: Path, training_device: str) -> None:
    directory = tmp_path / "input"
    prepare_input(directory)
    path = directory / "config.yaml"
    config = yaml.safe_load(path.read_text())
    for axis in ("y", "z"):
        config["domain"]["space"][axis] = {"interval": [0, 1], "points": 3}
    config["domain"]["parameters"] = {"amplitude": {"interval": [0.1, 0.5], "points": 2}}
    config["training"]["batch_size"] = 13
    config["training"]["collocation"] = {"sampling": "lhs", "points": 29}
    config["runtime"]["device"] = training_device
    config["runtime"]["precision"] = "32-true"
    config["coefficients"] = {"base": {"file": "base.npz"}}
    config["boundaries"].extend(
        [
            {"kind": "heat_flux", "field": "T", "axis": "y", "conductivity": "base", "value": 0},
            {"kind": "neumann", "field": "T", "axis": "z", "value": 0},
        ]
    )
    x = np.array([0.0, 0.4, 1.0])
    np.savez(directory / "base.npz", x=x, values=1 + x)
    (directory / "functions.py").write_text(
        """import torch
from pinnstorch.api import Coordinates, Equation, Problem, TensorFields

def initial(c: Coordinates) -> TensorFields:
    return {"T": torch.sin(torch.pi*c.x)}

def heat(e: Equation) -> torch.Tensor:
    k = e.parameters["amplitude"] * e.coefficients["base"]
    return e.dt("T") - e.divergence({a: k*v for a,v in e.gradient("T").items()})

problem = Problem(fields=("T",), equations={"heat": heat}, initial=initial)
""",
        encoding="utf-8",
    )
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "train"
    result = launch(directory, output)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = output / "checkpoints/final.ckpt"
    state = torch.load(checkpoint, map_location="cpu")
    assert state["global_step"] > 0
    assert all(torch.isfinite(value).all() for value in state["state_dict"].values())
    predictions: list[NDArray[np.float32]] = []
    for index, device in enumerate(("cpu", DEVICE)):
        config["runtime"]["device"] = device
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        destination = tmp_path / f"predict-{index}"
        result = launch(directory, destination, "predict", str(checkpoint))
        assert result.returncode == 0, result.stdout + result.stderr
        values = loadmat(destination / "predictions.mat")["T"]
        assert np.isfinite(values).all()
        predictions.append(values)
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=2e-4, atol=2e-5)
