"""Transport/Burgers preserve the public soft-mode file lifecycle without heat assumptions."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import loadmat

from benchmarks.cases import CASES
from tests.integration.test_runner_files import launch

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("name", ["advection", "burgers"])
def test_nonheat_train_exact_load_predict_and_retrain(tmp_path: Path, name: str) -> None:
    case = CASES[name]
    task = tmp_path / "input"
    task.mkdir()
    (task / "functions.py").write_bytes(case.source.read_bytes())
    config = case.config("cpu", 3, 19)
    config["network"]["hidden_layers"] = [8, 8]
    config["domain"]["space"]["x"]["points"] = 4
    config["domain"]["time"]["points"] = 3
    config["training"].update(
        constraints="soft",
        hard_execution="eager",
        refinement={"steps": 2},
        batch_size=7,
        collocation={"sampling": "lhs", "points": 7},
    )
    if name == "advection":
        assert config["boundaries"] == [
            dict(kind="dirichlet", field="u", axis="x", side="lower", value_function="inflow")
        ]
    else:
        assert config["boundaries"] == [dict(kind="dirichlet", field="u", axis="x", value=0)]
    path = task / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    trained = launch(task, tmp_path / "train")
    assert trained.returncode == 0, trained.stdout + trained.stderr
    checkpoint = tmp_path / "train/checkpoints/final.ckpt"
    state = torch.load(checkpoint, map_location="cpu")
    assert state["hyper_parameters"]["constraint_kind"] == "soft"
    assert state["hyper_parameters"]["hard_execution"] == "eager"
    assert state["global_step"] == 3

    probe = deepcopy(config)
    probe["training"].update(steps=1, refinement={"steps": 0})
    probe["training"]["optimizer"].update(learning_rate=1e-300, weight_decay=0)
    path.write_text(yaml.safe_dump(probe))
    loaded = launch(task, tmp_path / "load-probe", "retrain", str(checkpoint))
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr
    restored = torch.load(tmp_path / "load-probe/checkpoints/final.ckpt", map_location="cpu")
    assert state["state_dict"].keys() == restored["state_dict"].keys()
    for key, value in state["state_dict"].items():
        torch.testing.assert_close(restored["state_dict"][key], value, rtol=0, atol=0)

    config["domain"]["space"]["x"]["points"] = 5
    config["domain"]["time"]["points"] = 6
    path.write_text(yaml.safe_dump(config))
    predicted = launch(task, tmp_path / "predict", "predict", str(checkpoint))
    assert predicted.returncode == 0, predicted.stdout + predicted.stderr
    values = loadmat(tmp_path / "predict/predictions.mat")
    assert values["u"].shape == (5, 6)
    assert np.isfinite(values["u"]).all()
    np.testing.assert_allclose(values["x"].reshape(-1), np.linspace(0, 1, 5))
    np.testing.assert_allclose(values["t"].reshape(-1), np.linspace(0, 0.5, 6), atol=1e-7)

    retrained = launch(task, tmp_path / "retrain", "retrain", str(checkpoint))
    assert retrained.returncode == 0, retrained.stdout + retrained.stderr
    refined = torch.load(tmp_path / "retrain/checkpoints/final.ckpt", map_location="cpu")
    assert any(
        not torch.equal(value, refined["state_dict"][key])
        for key, value in state["state_dict"].items()
        if key.startswith("net.")
    )
    assert refined["hyper_parameters"]["constraint_kind"] == "soft"
    assert refined["hyper_parameters"]["hard_execution"] == "eager"
