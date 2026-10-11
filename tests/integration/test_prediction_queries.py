"""Explicit parameter queries, checkpoint domains, and stable batched outputs."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import loadmat

from pinnstorch.checkpoint_contract import validate_checkpoint_domain
from pinnstorch.data import create_interval_data
from tests.integration.test_runner_files import ROOT, launch


def test_fixed_parameter_keeps_normalization_bounds() -> None:
    data = create_interval_data(
        [0, 1],
        [0, 1],
        3,
        2,
        parameters={"k": {"interval": [0.1, 0.5], "points": 3}},
        parameter_values={"k": [0.25]},
    )
    assert data.spatial[0].shape == (3, 1)
    np.testing.assert_array_equal(data.spatial[1], np.full((3, 1), 0.25))
    np.testing.assert_allclose(data.normalization_bounds[0], [0, 0.1, 0])
    np.testing.assert_allclose(data.normalization_bounds[1], [1, 0.5, 1])


@pytest.mark.parametrize(
    "values",
    [
        {"unknown": [1]},
        {"k": []},
        {"k": [float("nan")]},
        {"k": [0.2, 0.2]},
        {"k": [[0.2]]},
        {"k": [True]},
        {"k": ["0.2"]},
    ],
)
def test_invalid_queries(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        create_interval_data(
            [0, 1],
            [0, 1],
            3,
            2,
            parameters={"k": {"interval": [0.1, 0.5], "points": 3}},
            parameter_values=values,
        )


def test_checkpoint_domain_contract() -> None:
    checkpoint = {
        "hyper_parameters": {
            "input_names": ("x", "k", "t"),
            "input_bounds": ((0, 1), (0.1, 0.5), (0, 1)),
        }
    }
    assert validate_checkpoint_domain(checkpoint, ("x", "k", "t"), [(0, 1), (0.25, 0.25), (0, 1)])[
        1
    ] == (0.1, 0.5)
    for query in [
        [(-1, 1), (0.25, 0.25), (0, 1)],
        [(0, 1), (0.7, 0.7), (0, 1)],
        [(0, 1), (0.25, 0.25), (0, 2)],
    ]:
        with pytest.raises(ValueError, match="outside checkpoint range"):
            validate_checkpoint_domain(checkpoint, ("x", "k", "t"), query)
        validate_checkpoint_domain(checkpoint, ("x", "k", "t"), query, allow_extrapolation=True)
    with pytest.raises(ValueError, match="input names"):
        validate_checkpoint_domain(checkpoint, ("x", "other", "t"), [(0, 1), (0.2, 0.2), (0, 1)])


def test_checkpoint_buffer_fallback() -> None:
    checkpoint = {
        "hyper_parameters": {"input_names": ("x", "k", "t")},
        "state_dict": {"net.lb": torch.tensor([0, 0.1, 0]), "net.ub": torch.tensor([1, 0.5, 1])},
    }
    validate_checkpoint_domain(checkpoint, ("x", "k", "t"), [(0, 1), (0.1, 0.5), (0, 1)])


def test_runner_parameter_queries(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("config.yaml", "functions.py"):
        shutil.copyfile(ROOT / "examples/heat_parameterized" / name, input_dir / name)
    cfg = yaml.safe_load((input_dir / "config.yaml").read_text())
    cfg["data"] = {"batch_size": 16}
    cfg["prediction"] = {"parameters": {"conductivity": [0.25]}}
    config_path = input_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    train_dir = tmp_path / "train"
    result = launch(input_dir, train_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = train_dir / "checkpoints/final.ckpt"
    state = torch.load(checkpoint, map_location="cpu")
    assert state["hyper_parameters"]["input_bounds"][2] == (0.1, 0.5)

    def predict(name: str) -> tuple[subprocess.CompletedProcess[str], Path]:
        directory = tmp_path / name
        result = launch(input_dir, directory, "predict", str(checkpoint))
        return result, directory

    result, directory = predict("fixed")
    assert result.returncode == 0, result.stdout + result.stderr
    saved = loadmat(directory / "predictions.mat")
    assert saved["T"].shape == (48, 3)
    np.testing.assert_array_equal(saved["conductivity"], np.full((48, 1), 0.25))
    metadata = json.loads((directory / "prediction.json").read_text())
    assert metadata["field_shape"] == [48, 3]
    assert metadata["batch_size"] == 16
    assert (
        yaml.safe_load((directory / "config.resolved.yaml").read_text())["data"]["batch_size"]
        == 16
    )
    assert metadata["training_bounds"][2] == [0.1, 0.5]
    assert metadata["requested_bounds"][2] == [0.25, 0.25]

    # Compare unbatched output and a deliberately non-sorted query grid.
    cfg["data"]["batch_size"] = None
    cfg["prediction"]["parameters"]["conductivity"] = [0.4, 0.25, 0.2]
    config_path.write_text(yaml.safe_dump(cfg))
    result, directory = predict("multiple")
    assert result.returncode == 0, result.stdout + result.stderr
    multiple = loadmat(directory / "predictions.mat")
    np.testing.assert_allclose(
        multiple["T"].reshape(48, 3, 3)[:, 1, :], saved["T"], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(multiple["conductivity"].reshape(48, 3)[0], [0.4, 0.25, 0.2])

    cfg["prediction"]["parameters"]["conductivity"] = [0.8]
    config_path.write_text(yaml.safe_dump(cfg))
    result, directory = predict("outside")
    assert result.returncode == 1
    assert "outside checkpoint range" in (directory / "error.log").read_text()
    assert not (directory / "predictions.mat").exists()
    cfg["prediction"]["allow_extrapolation"] = True
    config_path.write_text(yaml.safe_dump(cfg))
    result, directory = predict("intentional-extrapolation")
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_allclose(loadmat(directory / "predictions.mat")["conductivity"], 0.8)


def test_training_without_validation(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("config.yaml", "functions.py"):
        shutil.copyfile(ROOT / "examples/heat2d" / name, input_dir / name)
    cfg = yaml.safe_load((input_dir / "config.yaml").read_text())
    cfg["val_dataset"] = None
    cfg["callbacks"]["model_checkpoint"]["monitor"] = "train/loss"
    (input_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    output_dir = tmp_path / "train"
    result = launch(input_dir, output_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "checkpoints/final.ckpt").exists()
    assert (output_dir / "predictions.pkl").exists()
