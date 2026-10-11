"""Run independently of training fixtures: pytest --noconftest tests/runner."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[2] / "run.py"
spec = importlib.util.spec_from_file_location("runner_under_test", RUNNER_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_import_without_training_dependencies_or_valid_cli() -> None:
    code = f"""
import importlib.abc
import sys

class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'pinnstorch', 'torch', 'hydra', 'omegaconf'}}:
            raise AssertionError('Unexpected training import: ' + fullname)

sys.meta_path.insert(0, BlockTrainingImports())
sys.path.insert(0, {str(RUNNER_PATH.parent)!r})
sys.argv = ['run.py', '--unknown-test-argument']
import run
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", ["train", "retrain", "predict"])
def test_main_dispatches_mode_and_returns_exit_code(mode: str) -> None:
    with patch.object(runner, "run", return_value=7) as execute:
        assert runner.main(["--mode", mode]) == 7
    execute.assert_called_once_with(
        runner.Mode(mode), input_dir=runner.INPUT_DIR, output_dir=runner.OUTPUT_DIR
    )


@pytest.mark.parametrize("argv", [[], ["--mode", "invalid"]])
def test_invalid_arguments_do_not_start_training(argv: list[str]) -> None:
    with patch.object(runner, "run") as execute:
        with pytest.raises(SystemExit) as error:
            runner.main(argv)
    assert error.value.code == 2
    execute.assert_not_called()


def test_help_without_training_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--mode {train,retrain,predict}" in result.stdout


def test_main_passes_custom_directories() -> None:
    with patch.object(runner, "run", return_value=0) as execute:
        assert runner.main(
            ["--mode", "train", "--input-dir", "input", "--output-dir", "output"]
        ) == 0
    execute.assert_called_once_with(
        runner.Mode.TRAIN, input_dir=Path("input"), output_dir=Path("output")
    )
