"""Run the project's read-only quality checks in the current Python environment."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "run.py",
    "pinnstorch",
    "scripts",
    "tests/runner",
    "tests/numerics",
    "tests/integration",
    "examples/heat2d",
    "examples/heat_parameterized",
]


def main() -> int:
    checks = [
        [sys.executable, "-m", "ruff", "check", *TARGETS],
        [sys.executable, "-m", "ruff", "format", "--check", *TARGETS],
        [sys.executable, "-m", "mypy"],
        [sys.executable, "-m", "pytest", "-q"],
    ]
    for command in checks:
        print("Running: " + " ".join(command), flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
