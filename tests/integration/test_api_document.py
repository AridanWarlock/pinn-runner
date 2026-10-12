"""The complete YAML/Python blocks in API.md must remain executable."""

import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from tests.integration.test_runner_files import ROOT, launch

pytestmark = pytest.mark.e2e


def test_api_example(tmp_path: Path) -> None:
    source = (ROOT / "docs/API.md").read_text()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for marker, language, filename in (
        ("executable-config", "yaml", "config.yaml"),
        ("executable-functions", "python", "functions.py"),
    ):
        block = re.search(rf"<!-- {marker} -->\s*```{language}\n(.*?)\n```", source, re.DOTALL)
        assert block is not None
        (input_dir / filename).write_text(block.group(1) + "\n")
    train_dir = tmp_path / "train"
    result = launch(input_dir, train_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = train_dir / "checkpoints/final.ckpt"
    assert checkpoint.exists()
    predict_dir = tmp_path / "predict"
    result = launch(input_dir, predict_dir, "predict", str(checkpoint))
    assert result.returncode == 0, result.stdout + result.stderr
    saved = loadmat(predict_dir / "predictions.mat")
    assert saved["T"].shape == (48, 3)
    np.testing.assert_array_equal(saved["conductivity"], np.full((48, 1), 0.25))
    metadata = json.loads((predict_dir / "prediction.json").read_text())
    assert metadata["input_names"] == ["x", "y", "conductivity", "t"]
    assert metadata["batch_size"] == 16
    assert metadata["allow_extrapolation"] is False
