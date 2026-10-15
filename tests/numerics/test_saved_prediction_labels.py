"""Saved endpoint coordinates must reproduce the model, including a simulated time shift."""

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import savemat

from benchmarks.cases import CASES
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
    factory = SolverFactory(
        TaskSettings.read(config), load_problem(case.source), tmp_path, case.source.parent
    )
    model = factory.build(True).model.eval()
    mesh = torch.meshgrid(
        *(torch.linspace(0, 1, 3), torch.linspace(0, 1, 3), torch.linspace(0, 0.5, 3)),
        indexing="ij",
    )
    columns = [x.reshape(-1, 1) for x in mesh]
    with torch.no_grad():
        values = model(columns[:-1], columns[-1])["u"].numpy()
    if shift_time:
        values = np.roll(values.reshape(-1, 3), 1, axis=-1).reshape(-1, 1)
    (tmp_path / "prediction").mkdir()
    savemat(
        tmp_path / "prediction/predictions.mat",
        {"x": columns[0].numpy(), "y": columns[1].numpy(), "t": columns[2].numpy(), "u": values},
    )
    if shift_time:
        with pytest.raises(AssertionError):
            verify_saved_prediction(tmp_path, model, ("x", "y"), ("u",))
    else:
        report = verify_saved_prediction(tmp_path, model, ("x", "y"), ("u",))
        assert report["saved_prediction_endpoint_points_checked"] == 18
        assert report["saved_prediction_max_difference"] < 1e-6
