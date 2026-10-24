"""Физические часы: публичный файловый цикл и защита сохранённого правила."""

import hashlib
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import loadmat

from benchmarks.quadratic_clock_case import SOURCE, clock_config
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings
from tests.integration.test_runner_files import launch

pytestmark = pytest.mark.e2e


def test_scaled_time_public_lifecycle(tmp_path: Path) -> None:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    task = tmp_path / "input"
    task.mkdir()
    source = SOURCE
    original = source.read_bytes()
    (task / "functions.py").write_bytes(original)
    config = clock_config(device, 3, 11)
    config["domain"]["space"]["x"]["points"] = 4
    config["domain"]["time"]["points"] = 3
    config["training"].update(
        batch_size=7, collocation={"sampling": "lhs", "points": 7}, refinement={"steps": 2}
    )
    path = task / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
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
    assert saved["hyper_parameters"]["input_names"] == ("x", "a", "b", "t")
    identity = saved["hyper_parameters"]["input_transform"]
    assert identity["reference"] == 0.15
    assert identity["network_names"] == ["x", "b", "s"]
    assert saved["hyper_parameters"]["input_bounds"][1] == (0.05, 0.15)
    assert saved["global_step"] == 3
    config["domain"]["space"]["x"]["points"] = 5
    config["domain"]["time"]["points"] = 6
    config["prediction"] = {"parameter_rows": [[0.075, 1.4], [0.125, 1.6]]}
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    predicted = launch(task, tmp_path / "predict", "predict", str(checkpoint))
    assert predicted.returncode == 0, predicted.stdout + predicted.stderr
    values = loadmat(tmp_path / "predict/predictions.mat")
    settings = TaskSettings.read(config)
    np.testing.assert_array_equal(
        values["a"],
        settings.domain.grid(parameter_rows=settings.prediction.parameter_rows).spatial[1],
    )
    np.testing.assert_array_equal(
        values["b"],
        settings.domain.grid(parameter_rows=settings.prediction.parameter_rows).spatial[2],
    )
    assert values["u"].shape == (10, 6)
    factory = SolverFactory(
        settings, load_problem(task / "functions.py"), tmp_path / "direct", task
    )
    objects = factory.build(predicting=True)
    assert sum(p.numel() for p in objects.model.parameters()) == 2273
    objects.model.load_state_dict(saved["state_dict"], strict=True)
    spatial, clock = objects.mesh.collection_points(None, use_lhs=False)
    with torch.no_grad():
        direct = objects.model(
            [torch.tensor(spatial[:, i : i + 1], dtype=torch.float32) for i in range(3)],
            torch.tensor(clock, dtype=torch.float32),
        )["u"].numpy()
    np.testing.assert_allclose(values["u"].reshape(-1, 1), direct, atol=1e-6, rtol=1e-5)
    probe = deepcopy(config)
    probe["training"].update(steps=1, epochs=1, refinement={"steps": 0})
    probe["training"]["optimizer"].update(learning_rate=1e-300, weight_decay=0)
    path.write_text(yaml.safe_dump(probe, sort_keys=False))
    restored = launch(task, tmp_path / "restore", "retrain", str(checkpoint))
    assert restored.returncode == 0, restored.stdout + restored.stderr
    replay = torch.load(tmp_path / "restore/checkpoints/final.ckpt", map_location="cpu")
    for key, value in saved["state_dict"].items():
        torch.testing.assert_close(replay["state_dict"][key], value, rtol=0, atol=0)
    probe["prediction"]["parameter_rows"] = [[0.2, 1.5]]
    path.write_text(yaml.safe_dump(probe, sort_keys=False))
    rejected = launch(task, tmp_path / "outside", "predict", str(checkpoint))
    assert rejected.returncode == 1
    assert "outside checkpoint range" in (tmp_path / "outside/error.log").read_text()
    probe["prediction"]["parameter_rows"] = [[0.075, 1.4]]
    path.write_text(yaml.safe_dump(probe, sort_keys=False))
    (task / "functions.py").write_bytes(original.replace(b"** 2", b"** 3"))
    changed = launch(task, tmp_path / "changed", "predict", str(checkpoint))
    assert changed.returncode == 1
    assert "coefficient definitions differ" in (tmp_path / "changed/error.log").read_text()
    (task / "functions.py").write_bytes(original)
    probe["domain"]["parameters"] = {key: probe["domain"]["parameters"][key] for key in ("b", "a")}
    probe["domain"]["parameter_rows"] = [row[::-1] for row in probe["domain"]["parameter_rows"]]
    probe["prediction"]["parameter_rows"] = [[1.4, 0.075]]
    path.write_text(yaml.safe_dump(probe, sort_keys=False))
    wrong_order = launch(task, tmp_path / "wrong-order", "predict", str(checkpoint))
    assert wrong_order.returncode == 1
    assert "Checkpoint input transform" in (tmp_path / "wrong-order/error.log").read_text()
    for label, reference in (("wrong-reference", 0.2), ("missing-rule", None)):
        candidate = deepcopy(config)
        candidate["network"]["time_scale"] = (
            None if reference is None else {"parameter": "a", "reference": reference}
        )
        path.write_text(yaml.safe_dump(candidate, sort_keys=False))
        failed = launch(task, tmp_path / label, "predict", str(checkpoint))
        assert failed.returncode == 1
        assert "Checkpoint input transform" in (tmp_path / label / "error.log").read_text()
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    damaged = deepcopy(saved)
    damaged["state_dict"]["net.network.ub"][1] += 1
    bad_checkpoint = task / "damaged.ckpt"
    torch.save(damaged, bad_checkpoint)
    failed = launch(task, tmp_path / "damaged", "predict", str(bad_checkpoint))
    assert failed.returncode == 1
    assert "normalization differs" in (tmp_path / "damaged/error.log").read_text()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum
