"""Actual saved shared model predicts new k values without updating weights."""

import hashlib
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import loadmat

from benchmarks.cases import CASES
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings
from tests.integration.test_runner_files import launch

pytestmark = pytest.mark.e2e


def test_shared_constant_public_lifecycle(tmp_path: Path) -> None:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    case = CASES["heat_constant_surrogate"]
    task = tmp_path / "input"
    task.mkdir()
    original = case.source.read_bytes()
    (task / "functions.py").write_bytes(original)
    config = case.config(device, 3, 11)
    config["domain"]["space"]["x"]["points"] = 4
    config["domain"]["time"]["points"] = 3
    config["training"].update(
        batch_size=7, collocation={"sampling": "lhs", "points": 7}, refinement={"steps": 2}
    )
    path = task / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    trained = launch(task, tmp_path / "train")
    assert trained.returncode == 0, trained.stdout + trained.stderr
    checkpoint = tmp_path / "train/checkpoints/final.ckpt"
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    locations = []

    def on_cpu(storage, location):
        locations.append(location)
        return storage

    saved = torch.load(checkpoint, map_location=on_cpu)
    assert locations
    assert any(location.startswith("cuda") for location in locations) == (device == "cuda")
    assert saved["hyper_parameters"]["input_names"] == ("x", "k", "t")
    assert saved["global_step"] == 3
    config["domain"]["space"]["x"]["points"] = 5
    config["domain"]["time"]["points"] = 6
    config["prediction"] = {"parameters": {"k": [0.1, 0.2]}}
    path.write_text(yaml.safe_dump(config))
    predicted = launch(task, tmp_path / "predict", "predict", str(checkpoint))
    assert predicted.returncode == 0, predicted.stdout + predicted.stderr
    values = loadmat(tmp_path / "predict/predictions.mat")
    settings = TaskSettings.read(config)
    np.testing.assert_array_equal(
        values["k"], settings.domain.grid(settings.prediction.parameters).spatial[1]
    )
    assert values["u"].shape == (10, 6)
    factory = SolverFactory(
        settings, load_problem(task / "functions.py"), tmp_path / "direct", task
    )
    objects = factory.build(predicting=True)
    objects.model.load_state_dict(saved["state_dict"], strict=True)
    spatial, clock = objects.mesh.collection_points(None, use_lhs=False)
    with torch.no_grad():
        direct = objects.model(
            [torch.tensor(spatial[:, i : i + 1], dtype=torch.float32) for i in range(2)],
            torch.tensor(clock, dtype=torch.float32),
        )["u"].numpy()
    np.testing.assert_allclose(values["u"].reshape(-1, 1), direct, atol=1e-6, rtol=1e-5)
    probe = deepcopy(config)
    probe["training"].update(steps=1, epochs=1, refinement={"steps": 0})
    probe["training"]["optimizer"].update(learning_rate=1e-300, weight_decay=0)
    path.write_text(yaml.safe_dump(probe))
    restored = launch(task, tmp_path / "restore", "retrain", str(checkpoint))
    assert restored.returncode == 0, restored.stdout + restored.stderr
    replay = torch.load(tmp_path / "restore/checkpoints/final.ckpt", map_location="cpu")
    for key, value in saved["state_dict"].items():
        torch.testing.assert_close(replay["state_dict"][key], value, rtol=0, atol=0)
    probe["prediction"]["parameters"]["k"] = [0.3]
    path.write_text(yaml.safe_dump(probe))
    rejected = launch(task, tmp_path / "outside", "predict", str(checkpoint))
    assert rejected.returncode == 1
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum
