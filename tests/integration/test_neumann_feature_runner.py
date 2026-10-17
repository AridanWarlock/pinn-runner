"""Feature representation belongs to checkpoint meaning, independently of grid resolution."""

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
@pytest.mark.parametrize(
    "case_name,encoding,execution",
    [
        ("heat_neumann_polynomial1d", "neumann_cosine", "eager"),
        ("heat_neumann_polynomial1d", "neumann_polynomial", "neumann_analytic"),
        ("heat_neumann_polynomial3d", "neumann_polynomial", "eager"),
        ("heat_neumann_polynomial3d", "neumann_polynomial", "neumann_analytic"),
        ("heat_neumann2d", "neumann_polynomial", "eager"),
        ("heat_neumann2d", "neumann_polynomial", "neumann_analytic"),
    ],
)
def test_feature_checkpoint_predict_retrain_and_representation_guard(
    case_name: str, encoding: str, execution: str, tmp_path: Path
) -> None:
    case = CASES[case_name]
    config = case.config(os.environ.get("PINN_TEST_DEVICE", "cpu"), 3, 19)
    config["network"].update(hidden_layers=[4, 4], spatial_encoding=encoding)
    config["training"].update(
        constraints="hard_initial_robin",
        hard_execution=execution,
        refinement={"steps": 2},
        batch_size=7,
        collocation={"sampling": "lhs", "points": 7},
    )
    for axis in config["domain"]["space"].values():
        axis["points"] = 3
    config["domain"]["time"]["points"] = 3
    (tmp_path / "functions.py").write_bytes(case.source.read_bytes())
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    result = launch(tmp_path, tmp_path / "train", "train")
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = tmp_path / "train/checkpoints/final.ckpt"
    state = torch.load(checkpoint, map_location="cpu")
    assert state["hyper_parameters"]["spatial_encoding"] == encoding
    assert state["hyper_parameters"]["hard_execution"] == execution
    for axis in config["domain"]["space"].values():
        axis["points"] = 4
    path.write_text(yaml.safe_dump(config))
    result = launch(tmp_path, tmp_path / "predict", "predict", str(checkpoint))
    assert result.returncode == 0, result.stdout + result.stderr
    values = loadmat(tmp_path / "predict/predictions.mat")["u"]
    assert values.size == 4**case.dimensions * 3 and np.isfinite(values).all()
    if execution == "neumann_analytic":
        config["training"]["hard_execution"] = "eager"
        path.write_text(yaml.safe_dump(config))
        control = launch(tmp_path, tmp_path / "predict-control", "predict", str(checkpoint))
        assert control.returncode == 0, control.stdout + control.stderr
        np.testing.assert_array_equal(
            values, loadmat(tmp_path / "predict-control/predictions.mat")["u"]
        )
        config["training"]["hard_execution"] = execution
        path.write_text(yaml.safe_dump(config))
    result = launch(tmp_path, tmp_path / "retrain", "retrain", str(checkpoint))
    assert result.returncode == 0, result.stdout + result.stderr
    config["network"]["spatial_encoding"] = "identity"
    config["training"]["hard_execution"] = "robin_grouped"
    path.write_text(yaml.safe_dump(config))
    result = launch(tmp_path, tmp_path / "invalid", "predict", str(checkpoint))
    assert result.returncode != 0
    assert "Checkpoint hard constraint definition differs" in result.stdout + result.stderr
