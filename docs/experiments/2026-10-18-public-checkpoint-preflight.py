"""CPU-only file-contract preflight for a future cold prediction measurement."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.cases import CASES  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    args.output.mkdir(parents=True)
    input_dir = args.output / "input"
    input_dir.mkdir()
    case = CASES["heat_neumann_polynomial3d"]
    config = case.config("cpu", 1, 19)
    config["network"].update(spatial_encoding="neumann_polynomial")
    config["training"].update(
        constraints="hard_initial_robin",
        hard_execution="eager",
        batch_size=7,
        collocation={"sampling": "lhs", "points": 7},
    )
    for axis in config["domain"]["space"].values():
        axis["points"] = 4
    config["domain"]["time"]["points"] = 3
    (input_dir / "functions.py").write_bytes(case.source.read_bytes())
    (input_dir / "config.yaml").write_text(yaml.safe_dump(config))
    settings = TaskSettings.load(input_dir / "config.yaml")
    torch.manual_seed(19)
    objects = SolverFactory(
        settings, load_problem(input_dir / "functions.py"), args.output, input_dir
    ).build(predicting=True)
    # A recognizable field proves that CLI prediction loads these saved weights.
    with torch.no_grad():
        objects.model.net.network.model.output.weight.zero_()
        objects.model.net.network.model.output.bias.fill_(0.25)
    state = objects.model.state_dict()
    raw = args.output / "raw.pt"
    converted = args.output / "prediction.ckpt"
    torch.save(state, raw)
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    torch.save(
        dict(
            state_dict=state,
            hyper_parameters=dict(objects.model.hparams),
            research_conversion=dict(source_sha256=raw_sha, optimizer_state_restored=False),
        ),
        converted,
    )
    loaded = torch.load(converted, map_location="cpu")["state_dict"]
    assert loaded.keys() == state.keys()
    assert all(torch.equal(loaded[k], v) for k, v in state.items())
    env = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(args.output / "matplotlib")
    )
    outcomes = []
    for name, checkpoint in (("raw", raw), ("converted", converted)):
        target = args.output / name
        command = [
            sys.executable,
            "-B",
            str(ROOT / "run.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(target),
            "--mode",
            "predict",
            "--checkpoint",
            str(checkpoint),
        ]
        with (args.output / (name + ".log")).open("w") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=90,
                check=False,
            )
        outcomes.append(dict(kind=name, exit_code=result.returncode))
    assert outcomes[0]["exit_code"] != 0
    assert (
        "Checkpoint hard constraint definition differs"
        in (args.output / "raw/error.log").read_text()
    )
    assert outcomes[1]["exit_code"] == 0
    values = loadmat(args.output / "converted/predictions.mat")["u"]
    spatial = np.array(list(itertools.product(np.linspace(0, 1, 4), repeat=3)))
    q = spatial**2 * (3 - 2 * spatial)
    expected = np.prod(q, axis=1)[:, None] + 0.25 * np.linspace(0, 1, 3)[None, :]
    np.testing.assert_allclose(values, expected, rtol=2e-6, atol=5e-7)
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == raw_sha
    report = dict(
        device="cpu",
        torch=torch.__version__,
        training_steps=0,
        optimizer_history_present=False,
        all_state_tensors_equal=True,
        raw_source_unchanged=True,
        cli_outcomes=outcomes,
        field_shape=list(values.shape),
        known_field_max_difference=float(np.max(np.abs(values - expected))),
        cold_windows_timing_claim=False,
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
