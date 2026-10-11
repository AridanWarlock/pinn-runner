"""Exercise Cartesian coordinates, full boundary faces and the real file runner."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import loadmat

from pinnstorch.data import BoundaryConditions, PointCloud, create_interval_data
from tests.integration.test_runner_files import launch, prepare_input


@pytest.mark.parametrize("dimension", [2, 3])
def test_full_faces(dimension: int) -> None:
    kwargs = {"y_interval": [-2.0, 2.0], "y_points": 3}
    if dimension == 3:
        kwargs.update(z_interval=[4.0, 5.0], z_points=4)
    data = create_interval_data([0, 1], [0, 0.5], 2, 2, **kwargs)
    mesh = PointCloud(lambda: data)
    spatial, time, _ = mesh.flatten_mesh(None)
    assert len(spatial) == 2 * 3 * (4 if dimension == 3 else 1) * 2
    for index, axis in enumerate(("x", "y", "z")[:dimension]):
        for location in ("lower", "upper", "both"):
            boundary = BoundaryConditions(
                mesh, [{"kind": "dirichlet", "field": "T", "axis": axis, "side": location}]
            )
            actual = np.column_stack(
                [column.numpy() for column in boundary.spatial_domain_sampled]
            )
            mask = np.zeros(len(spatial), dtype=bool)
            if location in ("lower", "both"):
                mask |= spatial[:, index] == mesh.lb[index]
            if location in ("upper", "both"):
                mask |= spatial[:, index] == mesh.ub[index]
            np.testing.assert_array_equal(actual, spatial[mask].astype(np.float32))
            np.testing.assert_array_equal(boundary.time_domain_sampled.numpy(), time[mask])


@pytest.mark.parametrize("dimension", [2, 3])
def test_multidimensional_runner(tmp_path: Path, dimension: int) -> None:
    input_dir = tmp_path / "input"
    prepare_input(input_dir)
    config_file = input_dir / "config.yaml"
    cfg = yaml.safe_load(config_file.read_text())
    cfg["domain"]["space"] = {
        "x": {"interval": [0, 1], "points": 3},
        "y": {"interval": [0, 1], "points": 4},
    }
    if dimension == 3:
        cfg["domain"]["space"]["z"] = {"interval": [0, 1], "points": 2}
    cfg["boundaries"] = [
        {"kind": "dirichlet", "field": "T", "axis": axis} for axis in cfg["domain"]["space"]
    ]
    config_file.write_text(yaml.safe_dump(cfg))
    (input_dir / "functions.py").write_text("""import torch
from pinnstorch.api import Coordinates, Equation, Problem, TensorFields

def initial(c: Coordinates) -> TensorFields:
    result = torch.ones_like(c.t)
    for coordinate in c.spatial.values():
        result = result * torch.sin(torch.pi * coordinate)
    return {"T": result}

def heat(e: Equation) -> torch.Tensor:
    return e.dt("T") - 0.3 * e.laplacian("T")

problem = Problem(fields=("T",), equations={"heat": heat}, initial=initial)
""")
    output_dir = tmp_path / "train"
    result = launch(input_dir, output_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    prediction_dir = tmp_path / "predict"
    result = launch(
        input_dir, prediction_dir, "predict", str(output_dir / "checkpoints/best.ckpt")
    )
    assert result.returncode == 0, result.stdout + result.stderr
    saved = loadmat(prediction_dir / "predictions.mat")
    from pinnstorch.runner.settings import TaskSettings

    expected = TaskSettings.read(cfg).domain.grid()
    for index, axis in enumerate(("x", "y", "z")[:dimension]):
        np.testing.assert_array_equal(saved[axis], expected.spatial[index])
    assert saved["T"].shape == (12 * (2 if dimension == 3 else 1), 3)
    assert np.isfinite(saved["T"]).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"y_interval": [0, 1]},
        {"y_points": 3},
        {"z_interval": [0, 1], "z_points": 3},
        {"y_interval": [1, 0], "y_points": 3},
        {"y_interval": [0, 1], "y_points": True},
    ],
)
def test_invalid_axes(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        create_interval_data([0, 1], [0, 1], 3, 3, **kwargs)


@pytest.mark.parametrize("axis, location", [("z", "both"), ("q", "both"), ("x", "left")])
def test_rejects_invalid_boundary(axis: str, location: str) -> None:
    mesh = PointCloud(lambda: create_interval_data([0, 1], [0, 1], 3, 3))
    with pytest.raises(ValueError):
        BoundaryConditions(
            mesh, [{"kind": "dirichlet", "field": "T", "axis": axis, "side": location}]
        )
