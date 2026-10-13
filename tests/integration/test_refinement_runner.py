"""Public train/checkpoint/predict includes the explicit refinement stage."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import loadmat

from benchmarks.cases import CASES


@pytest.mark.e2e
@pytest.mark.parametrize("hard", [False, True])
def test_staged_runner_saves_both_stages_and_predicts(tmp_path: Path, hard: bool) -> None:
    case = CASES["burgers"]
    task, output = tmp_path / "task", tmp_path / "output"
    task.mkdir()
    config = case.config("cpu", 3, 19)
    config["network"]["hidden_layers"] = [8, 8]
    config["training"]["collocation"]["points"] = 32
    config["training"]["refinement"] = {"steps": 2}
    if hard:
        config["training"]["constraints"] = "hard_initial_dirichlet"
    (task / "config.yaml").write_text(yaml.safe_dump(config))
    (task / "functions.py").write_bytes(case.source.read_bytes())
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "matplotlib"))
    base = [sys.executable, str(root / "run.py"), "--input-dir", str(task)]
    trained = subprocess.run(
        base + ["--mode", "train", "--output-dir", str(output)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr
    adam = torch.load(output / "checkpoints/adam.ckpt", map_location="cpu")
    final = torch.load(output / "checkpoints/final.ckpt", map_location="cpu")
    assert adam["global_step"] == final["global_step"] == 3
    assert "refinement_state" not in adam
    stage = final["refinement_state"]
    assert stage["completed"] and stage["optimizer"]["state"]
    assert stage["device"] == "cpu"
    assert stage["metrics"]["lbfgs_steps"] == 2
    assert stage["metrics"]["lbfgs_closure_evaluations"] >= 2
    assert any(
        not torch.equal(value, final["state_dict"][name])
        for name, value in adam["state_dict"].items()
        if name.startswith("net.")
    )
    metrics = json.loads((output / "training.performance.json").read_text())
    assert metrics["adam_model_evaluations"] == 3
    assert metrics["lbfgs_residual_points"] == metrics["lbfgs_closure_evaluations"] * 32
    prediction = tmp_path / "prediction"
    predicted = subprocess.run(
        base
        + [
            "--mode",
            "predict",
            "--output-dir",
            str(prediction),
            "--checkpoint",
            str(output / "checkpoints/final.ckpt"),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert predicted.returncode == 0, predicted.stdout + predicted.stderr
    assert np.isfinite(loadmat(prediction / "predictions.mat")["u"]).all()
