"""File task checks for formula/table coefficients and checkpoint identity."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import loadmat

from tests.integration.test_runner_files import launch, prepare_input

pytestmark = pytest.mark.e2e

FUNCTIONS = """import torch
from pinnstorch.api import Coordinates, Equation, Problem, TensorFields

def initial(c: Coordinates) -> TensorFields:
    return {"T": torch.sin(torch.pi*c.x)}

def heat(e: Equation) -> torch.Tensor:
    flux = {axis: e.coefficients["conductivity"] * derivative for axis, derivative in e.gradient("T").items()}
    return e.dt("T") - e.divergence(flux)

def coefficient(c: Coordinates) -> torch.Tensor:
    return 1 + c.x + c.y

problem = Problem(fields=("T",), equations={"heat": heat}, initial=initialCOEFFICIENTS)
"""


@pytest.mark.parametrize("kind", ["table", "formula"])
def test_coefficient_runner_and_identity(tmp_path: Path, kind: str) -> None:
    input_dir = tmp_path / "input"
    prepare_input(input_dir)
    path = input_dir / "config.yaml"
    config = yaml.safe_load(path.read_text())
    config["domain"]["space"]["y"] = {"interval": [0, 1], "points": 3}
    config["training"]["batch_size"] = 7
    # Test an arbitrary LHS location, not just coefficient nodes.
    config["training"]["collocation"] = {"sampling": "lhs", "points": 19}
    config["boundaries"] = [
        {"kind": "dirichlet", "field": "T", "axis": "x", "value": 0},
        {
            "kind": "heat_flux",
            "field": "T",
            "axis": "y",
            "conductivity": "conductivity",
            "value": 0,
        },
    ]
    if kind == "table":
        config["coefficients"] = {"conductivity": {"file": "conductivity.npz"}}
        x, y = np.array([0.0, 0.2, 1.0]), np.array([0.0, 0.6, 1.0])
        np.savez(input_dir / "conductivity.npz", x=x, y=y, values=1 + x[:, None] + y[None, :])
    path.write_text(yaml.safe_dump(config))
    source = FUNCTIONS.replace(
        "COEFFICIENTS", ', coefficients={"conductivity": coefficient}' if kind == "formula" else ""
    )
    (input_dir / "functions.py").write_text(source)
    output = tmp_path / "train"
    result = launch(input_dir, output)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = output / "checkpoints/final.ckpt"
    state = torch.load(checkpoint, map_location="cpu")
    assert set(state["hyper_parameters"]["coefficient_signature"]) == {"conductivity"}
    if kind == "table":
        assert "coefficients.tables.field_conductivity.values" in state["state_dict"]
    definition = json.loads((output / "problem.json").read_text())
    assert definition["coefficients"] == ["conductivity"]
    predict = tmp_path / "predict"
    result = launch(input_dir, predict, "predict", str(checkpoint))
    assert result.returncode == 0, result.stdout + result.stderr
    saved = loadmat(predict / "predictions.mat")
    assert saved["T"].shape == (24, 3)
    assert np.isfinite(saved["T"]).all()
    assert (
        json.loads((predict / "coefficients.json").read_text())["signatures"]
        == state["hyper_parameters"]["coefficient_signature"]
    )
    if kind == "table":
        # Removing PDE/initial execution during predict remains supported.
        (input_dir / "functions.py").write_text(
            source.replace(
                'return e.dt("T") - e.divergence(flux)',
                'raise AssertionError("PDE called during inference")',
            ).replace(
                'return {"T": torch.sin(torch.pi*c.x)}',
                'raise AssertionError("initial called during inference")',
            )
        )
        result = launch(input_dir, tmp_path / "no-physics-predict", "predict", str(checkpoint))
        assert result.returncode == 0, result.stdout + result.stderr
        np.savez(input_dir / "conductivity.npz", x=x, y=y, values=2 + x[:, None] + y[None, :])
    else:
        (input_dir / "functions.py").write_text(
            source.replace("return 1 + c.x + c.y", "return 2 + c.x + c.y")
        )
    rejected = tmp_path / "rejected"
    result = launch(input_dir, rejected, "predict", str(checkpoint))
    assert result.returncode == 1
    assert "coefficient definitions differ" in (rejected / "error.log").read_text()
    assert not (rejected / "predictions.mat").exists()
