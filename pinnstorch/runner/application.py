"""File task orchestration without CLI parsing or backend details."""

from __future__ import annotations

import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from pinnstorch.utils.pylogger import get_pylogger

from .artifacts import ArtifactWriter
from .execution import SolverExecution
from .factory import SolverFactory
from .problem import load_problem
from .settings import TaskSettings


@dataclass(frozen=True)
class TaskPaths:
    input: Path
    output: Path
    checkpoint: Path

    @classmethod
    def resolve(cls, input: Path, output: Path, checkpoint: Path | None) -> TaskPaths:
        input = input.resolve()
        return cls(input, output.resolve(), (input / (checkpoint or "checkpoint.ckpt")).resolve())

    def prepare(self, mode: str) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "checkpoints").mkdir(exist_ok=True)
        files = [self.input / "config.yaml", self.input / "functions.py"]
        if mode in ("predict", "retrain"):
            files.append(self.checkpoint)
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(f"Required task file is missing: {path}")


class TaskApplication:
    def __init__(self, paths: TaskPaths) -> None:
        self.paths = paths
        self.log: logging.Logger = get_pylogger("pinnstorch.runner", paths.output / "log.log")
        self.log.setLevel(logging.INFO)

    def run(self, mode: str) -> int:
        try:
            self.paths.prepare(mode)
            self._execute(mode)
            self.log.info("Results: %s", self.paths.output)
            return 0
        except Exception as error:
            self._report(error)
            return 1

    def _execute(self, mode: str) -> None:
        settings = TaskSettings.load(self.paths.input / "config.yaml")
        problem = load_problem(self.paths.input / "functions.py")
        artifacts = ArtifactWriter(self.paths.output, settings, problem)
        artifacts.save_definition()
        objects = SolverFactory(settings, problem, self.paths.output).build(mode == "predict")
        execution = SolverExecution(settings, objects, artifacts, self.log)
        if mode == "predict":
            execution.predict(self.paths.checkpoint)
        else:
            execution.train(self.paths.checkpoint if mode == "retrain" else None)

    def _report(self, error: Exception) -> None:
        details = traceback.format_exc()
        self.log.error("Task failed: %s\n%s", error, details)
        self.paths.output.mkdir(parents=True, exist_ok=True)
        (self.paths.output / "error.log").write_text(
            f"PINN task failed\n{error}\n\n{details}", encoding="utf-8"
        )
        print(f"Running error: {error}", file=sys.stderr)
