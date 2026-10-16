"""The background strategy can change across checkpoint/predict/retrain."""

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
@pytest.mark.parametrize("dimensions,execution", [(1, "eager"), (3, "robin_grouped")])
def test_static_background_checkpoint_and_grid_roundtrip(
    dimensions: int, execution: str, tmp_path: Path
) -> None:
    case = CASES[f"heat_neumann_polynomial{dimensions}d"]
    config = case.config(os.environ.get("PINN_TEST_DEVICE", "cpu"), 3, 19)
    config["network"]["hidden_layers"] = [4, 4]
    for axis in config["domain"]["space"].values():
        axis["points"] = 3
    config["domain"]["time"]["points"] = 3
    config["training"].update(
        constraints="hard_initial_robin",
        hard_execution=execution,
        hard_background="general",
        refinement={"steps": 2},
        batch_size=7,
        collocation={"sampling": "lhs", "points": 7},
    )
    (tmp_path / "functions.py").write_bytes(case.source.read_bytes())
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    checkpoint: Path | None = None
    for name, mode, background in (
        ("train", "train", "general"),
        ("reference", "predict", "general"),
        ("static", "predict", "static"),
        ("retrain", "retrain", "static"),
    ):
        config["training"]["hard_background"] = background
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
        result = launch(tmp_path, tmp_path / name, mode, str(checkpoint) if checkpoint else None)
        assert result.returncode == 0, result.stdout + result.stderr
        if name == "train":
            state = torch.load(tmp_path / "train/checkpoints/final.ckpt", map_location="cpu")
            assert state["hyper_parameters"].pop("hard_background") == "general"
            # Historical checkpoints did not record this implementation detail.
            checkpoint = tmp_path / "historical.ckpt"
            torch.save(state, checkpoint)
        elif name == "retrain":
            state = torch.load(tmp_path / "retrain/checkpoints/final.ckpt", map_location="cpu")
            assert state["hyper_parameters"]["hard_background"] == "static"
            assert state["refinement_state"]["completed"]
    np.testing.assert_allclose(
        loadmat(tmp_path / "reference/predictions.mat")["u"],
        loadmat(tmp_path / "static/predictions.mat")["u"],
        rtol=3e-5,
        atol=3e-5,
    )
    for axis in config["domain"]["space"].values():
        axis["points"] = 4
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    result = launch(tmp_path, tmp_path / "new-grid", "predict", str(checkpoint))
    assert result.returncode == 0, result.stdout + result.stderr
    assert loadmat(tmp_path / "new-grid/predictions.mat")["u"].size == 4**dimensions * 3
