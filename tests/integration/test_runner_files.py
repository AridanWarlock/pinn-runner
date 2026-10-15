"""Launch the real CLI with public task descriptions and inspect artifacts."""

import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures/heat"

pytestmark = pytest.mark.e2e


def prepare_input(directory: Path) -> None:
    directory.mkdir()
    for name in ("config.yaml", "functions.py"):
        shutil.copyfile(FIXTURE / name, directory / name)


def launch(
    input_dir: Path, output_dir: Path, mode: str = "train", checkpoint: str | None = None
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ, MPLCONFIGDIR=str(input_dir.parent / "matplotlib"))
    command = [
        sys.executable,
        str(ROOT / "run.py"),
        "--mode",
        mode,
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", checkpoint])
    return subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=90, check=False
    )


def test_runner_trains_and_predicts_from_checkpoint(tmp_path: Path) -> None:
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    prepare_input(input_dir)
    result = launch(input_dir, output_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Starting training!" in (output_dir / "log.log").read_text()
    for name in ("best", "last", "final"):
        assert (output_dir / f"checkpoints/{name}.ckpt").exists()
    with (output_dir / "predictions.pkl").open("rb") as stream:
        predictions = pickle.load(stream)
    assert predictions["T"].shape == (24, 1)
    assert torch.isfinite(predictions["T"]).all()
    assert list(output_dir.glob("csv/**/metrics.csv"))
    assert "_target_" not in (output_dir / "config.resolved.yaml").read_text()
    assert (output_dir / "problem.json").exists()

    shutil.copyfile(output_dir / "checkpoints/best.ckpt", input_dir / "trained.ckpt")
    functions = input_dir / "functions.py"
    source = functions.read_text().replace(
        'return eq.dt("T") - 0.3 * eq.laplacian("T")',
        'raise AssertionError("Inference must not call equation")',
    )
    source = source.replace(
        'return {"T": torch.sin(torch.pi * c.x)}',
        'raise AssertionError("Inference must not call initial")',
    )
    functions.write_text(source)
    (input_dir / "data.mat").write_bytes(b"Unused reference")
    directory = tmp_path / "predict"
    result = launch(input_dir, directory, "predict", "trained.ckpt")
    assert result.returncode == 0, result.stdout + result.stderr
    saved = loadmat(directory / "predictions.mat")
    np.testing.assert_array_equal(saved["x"], np.linspace(0, 1, 8).reshape(-1, 1))
    np.testing.assert_allclose(saved["T"], predictions["T"].numpy().reshape(8, 3))
    assert "Starting training!" not in (directory / "log.log").read_text()
    state = torch.load(input_dir / "trained.ckpt", map_location="cpu")
    state["state_dict"].pop("net.lb")
    torch.save(state, input_dir / "broken.ckpt")
    result = launch(input_dir, tmp_path / "broken", "predict", "broken.ckpt")
    assert result.returncode == 1
    assert "Missing key(s)" in (tmp_path / "broken/error.log").read_text()


def test_invalid_problem_export(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    prepare_input(input_dir)
    (input_dir / "functions.py").write_text("# Missing Problem\n")
    result = launch(input_dir, tmp_path / "output")
    assert result.returncode == 1
    assert "must export problem = Problem" in (tmp_path / "output/error.log").read_text()


def test_initial_function_is_required(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    prepare_input(input_dir)
    functions = input_dir / "functions.py"
    functions.write_text(functions.read_text().replace("initial=initial", "initial=None"))
    result = launch(input_dir, tmp_path / "output")
    assert result.returncode == 1
    assert "must be callable" in (tmp_path / "output/error.log").read_text()


def test_public_potential_transform_predicts_without_equations(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    prepare_input(input_dir)
    source = """import torch
from pinnstorch.api import Coordinates, Equation, Problem, TensorFields

def initial(c: Coordinates) -> TensorFields:
    return {"T": torch.sin(torch.pi * c.x)}

def heat(e: Equation) -> torch.Tensor:
    return e.dt("T") - 0.3 * e.laplacian("T")

def transform(e: Equation) -> TensorFields:
    return {"T": e.diff("potential", "x")}

problem = Problem(fields=("T",), network_fields=("potential",), equations={"heat": heat}, initial=initial, transform=transform)
"""
    (input_dir / "functions.py").write_text(source)
    train_dir = tmp_path / "train"
    result = launch(input_dir, train_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    source = source.replace(
        'return {"T": torch.sin(torch.pi * c.x)}',
        'raise AssertionError("initial called during inference")',
    ).replace(
        'return e.dt("T") - 0.3 * e.laplacian("T")',
        'raise AssertionError("equation called during inference")',
    )
    (input_dir / "functions.py").write_text(source)
    predict_dir = tmp_path / "predict"
    result = launch(input_dir, predict_dir, "predict", str(train_dir / "checkpoints/final.ckpt"))
    assert result.returncode == 0, result.stdout + result.stderr
    saved = loadmat(predict_dir / "predictions.mat")
    assert saved["T"].shape == saved["potential"].shape == (8, 3)
    assert np.isfinite(saved["T"]).all()


def test_runner_loads_foreign_callback_path_for_prediction_and_retraining(tmp_path: Path) -> None:
    import pathlib

    class ForeignPath:
        def __reduce__(self) -> tuple[type[Path], tuple[str]]:
            foreign = pathlib.PosixPath if os.name == "nt" else pathlib.WindowsPath
            value = "/foreign/checkpoints" if os.name == "nt" else r"C:\foreign\checkpoints"
            return foreign, (value,)

    input_dir = tmp_path / "input"
    prepare_input(input_dir)
    trained = tmp_path / "train"
    result = launch(input_dir, trained)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = torch.load(trained / "checkpoints/final.ckpt", map_location="cpu")
    callback = next(iter(checkpoint["callbacks"].values()))
    assert isinstance(callback["dirpath"], str)
    callback["dirpath"] = ForeignPath()
    path = tmp_path / "foreign.ckpt"
    torch.save(checkpoint, path)
    for mode in ("predict", "retrain"):
        result = launch(input_dir, tmp_path / mode, mode, str(path))
        assert result.returncode == 0, result.stdout + result.stderr
    fresh = torch.load(tmp_path / "retrain/checkpoints/final.ckpt", map_location="cpu")
    assert isinstance(next(iter(fresh["callbacks"].values()))["dirpath"], str)
