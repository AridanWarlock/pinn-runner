"""Parameterized forward problems through the public problem language."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import loadmat

from pinnstorch.api import Coordinates, Equation, Problem
from pinnstorch.data import InitialCondition, PointCloud, create_interval_data
from pinnstorch.runner.problem import ProblemAdapter
from pinnstorch.runner.settings import TaskSettings
from tests.integration.test_runner_files import launch, prepare_input


def test_parameter_sampling_and_callbacks() -> None:
    settings = TaskSettings.read(
        {
            "version": 1,
            "domain": {
                "space": {"x": {"interval": [0, 1], "points": 3}},
                "time": {"interval": [0, 1], "points": 2},
                "parameters": {
                    "k": {"interval": [0.1, 0.5], "points": 3},
                    "amplitude": {"interval": [1, 2], "points": 2},
                },
            },
        }
    )
    data = settings.domain.grid()
    mesh = PointCloud(lambda: data)
    assert mesh.spatial_dim == 1
    assert mesh.input_dim == 3
    assert mesh.parameter_names == ("k", "amplitude")
    assert mesh.spatial_domain_mesh.shape == (18, 2, 3)
    np.testing.assert_allclose(mesh.lb, [0, 0.1, 1, 0])
    np.testing.assert_allclose(mesh.ub, [1, 0.5, 2, 1])
    columns, times = mesh.collection_points(100, use_lhs=True)
    assert columns.shape == (100, 3) and times.shape == (100, 1)
    assert np.all(columns >= mesh.lb[:-1]) and np.all(columns <= mesh.ub[:-1])
    problem = Problem(
        fields=("T",),
        initial=lambda c: {"T": c.parameters["amplitude"]},
        equations={"heat": lambda e: e.coordinates.x + e.coordinates.t + e.parameters["k"]},
    )
    adapter = ProblemAdapter(problem, settings.domain)
    dataset = InitialCondition(mesh, solution=["T"], initial_fun=adapter.initial)
    np.testing.assert_array_equal(dataset[:][2]["T"].numpy(), data.spatial[2])
    spatial, time, _ = dataset[:]
    residual = adapter.residuals({"T": spatial[-1]}, *spatial, time)["heat"]
    torch.testing.assert_close(residual, spatial[0] + time + spatial[1])


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.e2e
def test_parameterized_runner(tmp_path: Path, dimension: int) -> None:
    input_dir = tmp_path / "input"
    prepare_input(input_dir)
    cfg = yaml.safe_load((input_dir / "config.yaml").read_text())
    cfg["domain"]["space"]["x"]["points"] = 3
    cfg["domain"]["parameters"] = {"conductivity": {"interval": [0.1, 0.5], "points": 3}}
    for axis in ("y", "z")[: dimension - 1]:
        cfg["domain"]["space"][axis] = {"interval": [0, 1], "points": 2}
    cfg["boundaries"] = [
        {"kind": "dirichlet", "field": "T", "axis": axis} for axis in cfg["domain"]["space"]
    ]
    (input_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    (input_dir / "functions.py").write_text("""import torch
from pinnstorch.api import Coordinates, Equation, Problem, TensorFields

def initial(c: Coordinates) -> TensorFields:
    result = c.parameters["conductivity"]
    for coordinate in c.spatial.values():
        result = result * torch.sin(torch.pi * coordinate)
    return {"T": result}

def heat(e: Equation) -> torch.Tensor:
    return e.dt("T") - e.parameters["conductivity"] * e.laplacian("T")

problem = Problem(fields=("T",), equations={"heat": heat}, initial=initial)
""")
    output_dir = tmp_path / "train"
    result = launch(input_dir, output_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    cfg["domain"]["parameters"]["conductivity"] = {"interval": [0.2, 0.4], "points": 2}
    (input_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    prediction_dir = tmp_path / "predict"
    checkpoint_path = output_dir / "checkpoints/best.ckpt"
    result = launch(input_dir, prediction_dir, "predict", str(checkpoint_path))
    assert result.returncode == 0, result.stdout + result.stderr
    saved = loadmat(prediction_dir / "predictions.mat")
    np.testing.assert_array_equal(
        saved["conductivity"], TaskSettings.read(cfg).domain.grid().spatial[-1]
    )
    assert saved["T"].shape == (3 * 2 ** (dimension - 1) * 2, 3)
    assert np.isfinite(saved["T"]).all()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["hyper_parameters"]["input_names"] == (
        *("x", "y", "z")[:dimension],
        "conductivity",
        "t",
    )
    if dimension == 1:
        result = launch(input_dir, tmp_path / "retrain", "retrain", str(checkpoint_path))
        assert result.returncode == 0, result.stdout + result.stderr
    cfg["domain"]["parameters"] = {"other_coefficient": {"interval": [0.2, 0.4], "points": 2}}
    (input_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    rejected = tmp_path / "wrong-parameter"
    result = launch(input_dir, rejected, "predict", str(checkpoint_path))
    assert result.returncode == 1
    assert "Checkpoint input names" in (rejected / "error.log").read_text()


@pytest.mark.parametrize(
    "parameters",
    [
        {"x": {"interval": [0, 1], "points": 2}},
        {"bad-name": {"interval": [0, 1], "points": 2}},
        {"k": {"interval": [0, 1]}},
        {"k": {"interval": [1, 0], "points": 2}},
        {"k": {"interval": [0, 1], "points": True}},
    ],
)
def test_invalid_parameters(parameters: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        create_interval_data([0, 1], [0, 1], 3, 3, parameters=parameters)


def test_output_transform_preserves_derivatives() -> None:
    c = Coordinates(
        {"x": torch.linspace(0.1, 0.9, 7).reshape(-1, 1).requires_grad_()},
        torch.zeros(7, 1),
        {"k": torch.linspace(0.1, 0.5, 7).reshape(-1, 1)},
    )
    equation = Equation({"potential": c.x.square()}, c)
    transformed = equation.field("potential") * equation.parameters["k"]
    torch.testing.assert_close(equation.diff(transformed, "x"), 2 * c.parameters["k"] * c.x)
