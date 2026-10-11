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
from scipy.io import savemat

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


def launch(input_dir: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ, MPLCONFIGDIR=str(input_dir.parent / "matplotlib"))
    return subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--mode", "train",
         "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_runner_trains_from_files_and_saves_predictions(tmp_path: Path) -> None:
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


def test_runner_reports_invalid_user_functions(tmp_path: Path) -> None:
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    prepare_input(input_dir)
    (input_dir / "functions.py").write_text("# Required functions are missing.\n")

    result = launch(input_dir, output_dir)

    assert result.returncode == 1
    assert "Required function 'read_data_fn' not found" in (output_dir / "error.log").read_text()
    assert not (output_dir / "predictions.pkl").exists()
