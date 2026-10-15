"""Use the real MAT writer to check endpoint labels and a simulated time shift."""

from pathlib import Path

import pytest
import torch

from benchmarks.cases import CASES
from pinnstorch.runner.artifacts import ArtifactWriter
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings
from scripts.benchmark_residual_slices import verify_saved_prediction


@pytest.mark.parametrize("shift_time", [False, True])
def test_saved_field_labels_are_checked_against_direct_model(
    tmp_path: Path, shift_time: bool
) -> None:
    case = CASES["heat_neumann2d"]
    config = case.config("cpu", 1, 19)
    config["training"].update(constraints="hard_initial_robin", hard_lifting="expanded")
    config["network"]["hidden_layers"] = [8, 8]
    for axis in config["domain"]["space"].values():
        axis["points"] = 3
    config["domain"]["time"]["points"] = 3
    settings = TaskSettings.read(config)
    problem = load_problem(case.source)
    factory = SolverFactory(settings, problem, tmp_path, case.source.parent)
    objects = factory.build(True)
    model = objects.model.eval()
    points, times, _ = objects.mesh.flatten_mesh(None)
    spatial = [
        torch.tensor(points[:, i : i + 1], dtype=torch.float32) for i in range(points.shape[1])
    ]
    clock = torch.tensor(times, dtype=torch.float32)
    with torch.no_grad():
        values = model(spatial, clock)["u"]
    if shift_time:
        values = torch.roll(values.reshape(-1, 3), 1, dims=-1).reshape(-1, 1)
    prediction = tmp_path / "prediction"
    prediction.mkdir()
    writer = ArtifactWriter(prediction, settings, problem)
    bounds = tuple(
        axis.interval for axis in (*settings.domain.space.values(), settings.domain.time)
    )
    writer.save_prediction({"u": values}, objects.mesh, bounds)
    if shift_time:
        with pytest.raises(AssertionError):
            verify_saved_prediction(tmp_path, model, ("x", "y"), ("u",))
    else:
        report = verify_saved_prediction(tmp_path, model, ("x", "y"), ("u",))
        assert report["saved_prediction_endpoint_points_checked"] == 18
        assert report["saved_prediction_max_difference"] < 1e-6
