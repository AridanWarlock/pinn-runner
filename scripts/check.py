"""Run the project's read-only quality checks in the current Python environment."""

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Show detailed pytest diagnostics")
    arguments = parser.parse_args()
    pytest_options: list[str] = (
        [
            "-o",
            "addopts=--strict-markers --color=auto",
            "-o",
            "log_cli=true",
            "-vv",
            "--tb=long",
            "--durations=10",
        ]
        if arguments.verbose
        else []
    )
    checks: list[tuple[str, list[str]]] = [
        ("Ruff lint", [sys.executable, "-m", "ruff", "check", *TARGETS]),
        ("Ruff format", [sys.executable, "-m", "ruff", "format", "--check", *TARGETS]),
        ("mypy", [sys.executable, "-m", "mypy"]),
        ("pytest", [sys.executable, "-m", "pytest", *pytest_options]),
    ]
    for label, command in checks:
        print(f"Running: {label}", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
