"""Result files retain a portable dtype independently of training precision."""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from pinnstorch.api import Problem
from pinnstorch.data import PointCloud
from pinnstorch.runner.artifacts import ArtifactWriter
from pinnstorch.runner.settings import TaskSettings


def test_half_precision_results_are_portable(tmp_path: Path) -> None:
    settings = TaskSettings.read(
        {
            "version": 1,
            "domain": {
                "space": {"x": {"interval": [0, 1], "points": 2}},
                "time": {"interval": [0, 1], "points": 2},
            },
        }
    )
    problem = Problem(
        fields=("T",),
        equations={"heat": lambda e: e.field("T")},
        initial=lambda c: {"T": torch.zeros_like(c.t)},
    )
    mesh = PointCloud(settings.domain.grid)
    writer = ArtifactWriter(tmp_path, settings, problem)
    predictions = {"T": torch.tensor([[1], [2], [3], [4]], dtype=torch.float16)}
    writer.save_prediction(predictions, mesh, ((0, 1), (0, 1)))
    writer.save_training_predictions(predictions)
    saved = loadmat(tmp_path / "predictions.mat")["T"]
    assert saved.dtype == np.float32
    np.testing.assert_array_equal(saved, [[1, 2], [3, 4]])
    metadata = json.loads((tmp_path / "prediction.json").read_text())
    assert metadata["dtype"] == str(saved.dtype)
    with (tmp_path / "predictions.pkl").open("rb") as stream:
        tensor = pickle.load(stream)["T"]
    assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
    torch.testing.assert_close(tensor, predictions["T"].float())
