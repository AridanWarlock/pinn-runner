"""Launch the real CLI using user input files and inspect its output artifacts."""

import math
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures/heat"


def prepare_input(directory: Path) -> None:
    directory.mkdir()
    for name in ("config.yaml", "functions.py"):
        shutil.copyfile(FIXTURE / name, directory / name)
    x = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    time = np.linspace(0.0, 0.5, 3).reshape(-1, 1)
    temperature = np.sin(math.pi * x) * np.exp(-0.3 * math.pi**2 * time.T)
    savemat(directory / "data.mat", {"x": x, "t": time, "T": temperature})


def launch(
    input_dir: Path, output_dir: Path, mode: str = "train", checkpoint: str | None = None
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ, MPLCONFIGDIR=str(input_dir.parent / "matplotlib"))
    command = [sys.executable, str(ROOT / "run.py"), "--mode", mode,
               "--input-dir", str(input_dir), "--output-dir", str(output_dir)]
    if checkpoint is not None:
        command.extend(["--checkpoint", checkpoint])
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_runner_trains_and_predicts_from_checkpoint(tmp_path: Path) -> None:
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    prepare_input(input_dir)

    result = launch(input_dir, output_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (output_dir / "error.log").exists()
    assert "Starting training!" in (output_dir / "log.log").read_text()
    assert (output_dir / "checkpoints/best.ckpt").is_file()
    assert (output_dir / "checkpoints/last.ckpt").is_file()
    assert list(output_dir.glob("csv/**/metrics.csv"))
    with (output_dir / "predictions.pkl").open("rb") as file:
        predictions = pickle.load(file)
    assert predictions["T"].shape == (24, 1)
    assert torch.isfinite(predictions["T"]).all()

    shutil.copyfile(output_dir / "checkpoints/best.ckpt", input_dir / "trained.ckpt")
    prediction_dir = tmp_path / "prediction"
    result = launch(input_dir, prediction_dir, mode="predict", checkpoint="trained.ckpt")

    assert result.returncode == 0, result.stdout + result.stderr
    saved = loadmat(prediction_dir / "predictions.mat")
    source = loadmat(input_dir / "data.mat")
    np.testing.assert_array_equal(saved["x"], source["x"])
    np.testing.assert_array_equal(saved["t"], source["t"])
    np.testing.assert_allclose(saved["T"], predictions["T"].numpy().reshape(8, 3))
    assert np.isfinite(saved["T"]).all()
    assert "Starting training!" not in (prediction_dir / "log.log").read_text()

    broken = torch.load(input_dir / "trained.ckpt", map_location="cpu")
    broken["state_dict"].pop(next(iter(broken["state_dict"])))
    incompatible = input_dir / "incompatible.ckpt"
    torch.save(broken, incompatible)
    failed_dir = tmp_path / "incompatible-output"
    result = launch(input_dir, failed_dir, mode="predict", checkpoint=str(incompatible))

    assert result.returncode == 1
    assert "Missing key(s) in state_dict" in (failed_dir / "error.log").read_text()
    assert not (failed_dir / "predictions.mat").exists()


def test_runner_reports_invalid_user_functions(tmp_path: Path) -> None:
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    prepare_input(input_dir)
    (input_dir / "functions.py").write_text("# Required functions are missing.\n")

    result = launch(input_dir, output_dir)

    assert result.returncode == 1
    assert "Required function 'read_data_fn' not found" in (output_dir / "error.log").read_text()
    assert not (output_dir / "predictions.pkl").exists()


def test_predict_rejects_missing_data_before_loading_checkpoint(tmp_path: Path) -> None:
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    prepare_input(input_dir)
    (input_dir / "data.mat").unlink()
    (input_dir / "checkpoint.ckpt").write_bytes(b"Not a real checkpoint")

    result = launch(input_dir, output_dir, mode="predict")

    assert result.returncode == 1
    assert "Missing:" in result.stderr and "data.mat" in result.stderr
    assert not (output_dir / "predictions.mat").exists()
