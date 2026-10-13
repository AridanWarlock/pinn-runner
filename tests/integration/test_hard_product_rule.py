"""Public train/refine/checkpoint/predict roundtrip for the hard derivative backend."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import loadmat

from benchmarks.cases import CASES
from tests.integration.test_runner_files import launch


@pytest.mark.e2e
def test_product_rule_train_refine_and_eager_prediction(tmp_path: Path) -> None:
    case = CASES["heat2d"]
    states = []
    for backend in ("eager", "product_rule"):
        directory = tmp_path / backend
        directory.mkdir()
        config = case.config("cpu", 3, 19)
        config["training"].update(
            constraints="hard_initial_dirichlet",
            hard_execution=backend,
            refinement={"steps": 2},
            batch_size=17,
            collocation={"sampling": "lhs", "points": 17},
        )
        (directory / "functions.py").write_bytes(case.source.read_bytes())
        (directory / "config.yaml").write_text(yaml.safe_dump(config))
        output = directory / "trained"
        result = launch(directory, output)
        assert result.returncode == 0, result.stdout + result.stderr
        checkpoint = output / "checkpoints/final.ckpt"
        state = torch.load(checkpoint, map_location="cpu")
        assert state["hyper_parameters"]["hard_execution"] == backend
        assert state["refinement_state"]["metrics"]["lbfgs_steps"] == 2
        states.append(state["state_dict"])
        # Runtime recipe can change without changing the mathematical checkpoint.
        config["training"]["hard_execution"] = "eager"
        (directory / "config.yaml").write_text(yaml.safe_dump(config))
        result = launch(directory, directory / "predicted", "predict", str(checkpoint))
        assert result.returncode == 0, result.stdout + result.stderr
        assert np.isfinite(loadmat(directory / "predicted/predictions.mat")["u"]).all()
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=2e-4, atol=2e-5)
