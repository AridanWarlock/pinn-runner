"""Public interval Robin/Neumann train, refinement, checkpoint, predict and fine-tune."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import loadmat

from benchmarks.cases import CASES
from tests.integration.test_runner_files import launch


@pytest.mark.e2e
@pytest.mark.parametrize("name", ["heat_neumann", "heat_robin", "heat_neumann2d", "heat_robin3d"])
def test_robin_file_roundtrip(tmp_path: Path, name: str) -> None:
    case = CASES[name]
    config = case.config(os.environ.get("PINN_TEST_DEVICE", "cpu"), 3, 19)
    config["training"].update(
        constraints="hard_initial_robin",
        refinement={"steps": 2},
        batch_size=17,
        collocation={"sampling": "lhs", "points": 17},
    )
    if case.dimensions > 1:
        config["network"]["hidden_layers"] = [8, 8]
        for axis in config["domain"]["space"].values():
            axis["points"] = 3
        config["domain"]["time"]["points"] = 3
    (tmp_path / "functions.py").write_bytes(case.source.read_bytes())
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    for mode in ("train", "predict", "retrain"):
        output = tmp_path / mode
        checkpoint = None if mode == "train" else str(tmp_path / "train/checkpoints/final.ckpt")
        result = launch(tmp_path, output, mode, checkpoint)
        assert result.returncode == 0, result.stdout + result.stderr
        if mode == "predict":
            assert np.isfinite(loadmat(output / "predictions.mat")["u"]).all()
        else:
            state = torch.load(output / "checkpoints/final.ckpt", map_location="cpu")
            assert state["hyper_parameters"]["constraint_kind"] == "hard_initial_robin"
            assert state["refinement_state"]["device"].split(":")[0] == config["runtime"]["device"]
    config["boundaries"][0]["weight"] = 2
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    result = launch(
        tmp_path,
        tmp_path / "same-physics",
        "predict",
        str(tmp_path / "train/checkpoints/final.ckpt"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
