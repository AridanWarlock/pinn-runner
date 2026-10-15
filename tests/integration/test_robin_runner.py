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
@pytest.mark.parametrize("backend", ["eager", "robin_batch", "robin_grouped"])
@pytest.mark.parametrize("name", ["heat_neumann", "heat_robin", "heat_neumann2d", "heat_robin3d"])
def test_robin_file_roundtrip(
    tmp_path: Path,
    name: str,
    backend: str,
    lifting: str = "recursive",
    time_power: float = 0,
    backward: str = "all",
) -> None:
    case = CASES[name]
    config = case.config(os.environ.get("PINN_TEST_DEVICE", "cpu"), 3, 19)
    config["training"].update(
        constraints="hard_initial_robin",
        hard_execution=backend,
        hard_lifting=lifting,
        backward=backward,
        refinement={"steps": 2},
        batch_size=17,
        collocation={"sampling": "lhs", "points": 17},
    )
    if time_power:
        config["training"]["collocation"].update(time_power=time_power, resample_every_epochs=1)
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
            assert state["hyper_parameters"]["hard_execution"] == backend
            assert state["hyper_parameters"]["hard_lifting"] == lifting
            assert state["hyper_parameters"]["backward_mode"] == (
                "parameters" if backward == "auto" else backward
            )
            assert state["refinement_state"]["device"].split(":")[0] == config["runtime"]["device"]
    config["boundaries"][0]["weight"] = 2
    config["training"]["hard_execution"] = "robin_batch" if backend == "eager" else "eager"
    config["training"]["hard_lifting"] = "expanded" if lifting == "recursive" else "recursive"
    config["training"]["backward"] = "all" if backward in ("parameters", "auto") else "parameters"
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    result = launch(
        tmp_path,
        tmp_path / "same-physics",
        "predict",
        str(tmp_path / "train/checkpoints/final.ckpt"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_allclose(
        loadmat(tmp_path / "same-physics/predictions.mat")["u"],
        loadmat(tmp_path / "predict/predictions.mat")["u"],
        rtol=3e-5,
        atol=3e-5,
    )


@pytest.mark.e2e
@pytest.mark.parametrize("name", ["heat_neumann3d", "heat_mixed3d"])
def test_expanded_robin_file_roundtrip(tmp_path: Path, name: str) -> None:
    test_robin_file_roundtrip(tmp_path, name, "robin_grouped", "expanded")


@pytest.mark.e2e
@pytest.mark.parametrize("name", ["heat_robin", "heat_mixed3d"])
def test_time_biased_robin_file_roundtrip(tmp_path: Path, name: str) -> None:
    test_robin_file_roundtrip(tmp_path, name, "robin_grouped", time_power=1)
    state = torch.load(tmp_path / "train/checkpoints/final.ckpt", map_location="cpu")
    fixed = state["refinement_state"]["fixed_batches"]["equations"][1]
    restored = state["PINNDataModule"]["equations"]
    assert restored["signature"][-2:] == ("time_power", 1)
    torch.testing.assert_close(fixed, restored["time"], rtol=0, atol=0)


@pytest.mark.e2e
@pytest.mark.parametrize("name", ["heat_robin", "heat_mixed3d"])
def test_parameter_backward_file_roundtrip(tmp_path: Path, name: str) -> None:
    test_robin_file_roundtrip(tmp_path, name, "robin_grouped", "expanded", backward="parameters")


@pytest.mark.e2e
def test_auto_backward_file_roundtrip(tmp_path: Path) -> None:
    test_robin_file_roundtrip(tmp_path, "heat_neumann2d", "robin_grouped", backward="auto")
