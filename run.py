#!/usr/bin/env python3
"""Import-safe CLI for one disposable PINN task."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

INPUT_DIR = Path("/task_data")
OUTPUT_DIR = Path("/task_output")


class Mode(Enum):
    TRAIN = "train"
    RETRAIN = "retrain"
    PREDICT = "predict"


class RunnerArguments(argparse.Namespace):
    """Arguments accepted by the container entry point."""

    mode: str
    checkpoint: Path | None
    input_dir: Path
    output_dir: Path


def parse_args(argv: Sequence[str] | None = None) -> RunnerArguments:
    """Parse CLI arguments only when explicitly called."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=[mode.value for mode in Mode],
        help="Execution mode (train, retrain, predict)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file containing trained model weights",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=INPUT_DIR, help="Directory with task files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for results"
    )
    return parser.parse_args(argv, namespace=RunnerArguments())


def run(
    mode: Mode,
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    checkpoint: Path | None = None,
) -> int:
    from pinnstorch.runner.application import TaskApplication, TaskPaths

    paths = TaskPaths.resolve(input_dir, output_dir, checkpoint)
    paths.output.mkdir(parents=True, exist_ok=True)
    return TaskApplication(paths).run(mode.value)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    return run(
        Mode(arguments.mode),
        input_dir=arguments.input_dir,
        output_dir=arguments.output_dir,
        checkpoint=arguments.checkpoint,
    )


if __name__ == "__main__":
    sys.exit(main())
