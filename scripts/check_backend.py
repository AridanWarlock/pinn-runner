"""Run explicit CPU/CUDA acceptance and retain environment, log and JUnit reports."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("backend-report"))
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    working_tree = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    environment: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": arguments.device,
        "revision": revision.stdout.strip(),
        "working_tree_dirty": bool(working_tree.stdout.strip()),
        "precision": "32-true",
        "packages": {name: version(name) for name in ("numpy", "lightning", "pytest")},
    }
    if arguments.device == "cuda" and not torch.cuda.is_available():
        environment.update(status="unavailable", exit_code=2)
        (output / "environment.json").write_text(
            json.dumps(environment, indent=2), encoding="utf-8"
        )
        print("CUDA unavailable: see environment.json. No acceptance tests ran.")
        return 2
    if arguments.device == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(0)
        environment["compute_capability"] = torch.cuda.get_device_capability(0)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/hardware",
        "-vv",
        "--junitxml",
        str(output / "results.xml"),
    ]
    print(f"Running {arguments.device} acceptance; log: {output / 'tests.log'}", flush=True)
    with (output / "tests.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(os.environ, PINN_TEST_DEVICE=arguments.device),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    environment.update(
        status="passed" if result.returncode == 0 else "failed", exit_code=result.returncode
    )
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(f"Acceptance {environment['status']}; exit code {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
