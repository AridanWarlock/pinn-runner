"""Numerical PDE acceptance: train, evaluate against analytics, compare a baseline."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

# Also allow `python scripts/benchmark_solutions.py` from a fresh clone.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.cases import CASES  # noqa: E402
from benchmarks.execution import run_trial  # noqa: E402
from benchmarks.metrics import regression_failures  # noqa: E402


def environment(device: str) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return {
        "revision": revision.stdout.strip(),
        "working_tree_dirty": bool(dirty.stdout.strip()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "packages": {name: version(name) for name in ("numpy", "lightning")},
    }


def compare(report: dict[str, Any], path: Path) -> list[str]:
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if (
        baseline.get("contract") != report["contract"]
        or baseline.get("profile") != report["profile"]
    ):
        raise ValueError("Baseline contract/profile differs")
    if baseline.get("status") != "passed":
        raise ValueError("Baseline must have passed numerical acceptance")
    previous = {(item["case"], item["seed"]): item for item in baseline["trials"]}
    current = {(item["case"], item["seed"]): item for item in report["trials"]}
    if set(current) != set(previous):
        raise ValueError("Baseline cases/seeds differ")
    failures: list[str] = []
    for key, trial in current.items():
        old = previous[key]
        if trial["fingerprint"] != old["fingerprint"]:
            raise ValueError(f"Baseline task/budget/evaluation differs for {key}")
        failures.extend(
            f"{key}: {message}"
            for message in regression_failures(trial["metrics"], old["metrics"])
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--profile", choices=("smoke", "standard", "reference"), default="standard"
    )
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 4321])
    parser.add_argument("--steps", type=int, help="Override the fixed profile budget")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="A new directory for this run"
    )
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    steps = (
        args.steps
        if args.steps is not None
        else {"smoke": 3, "standard": 3000, "reference": 6000}[args.profile]
    )
    if steps < 1 or any(seed < 0 for seed in args.seeds):
        parser.error("steps must be positive and seeds nonnegative")
    if len(set(args.cases)) != len(args.cases) or len(set(args.seeds)) != len(args.seeds):
        parser.error("cases and seeds must be unique")
    if args.device == "cpu" and args.profile != "smoke":
        parser.error("Long numerical acceptance runs on Windows CUDA; CPU is for smoke only")
    if args.profile == "smoke" and args.baseline:
        parser.error("Smoke does not check quality and cannot compare baselines")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; no CPU fallback")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "contract": 1,
        "profile": args.profile,
        "environment": environment(args.device),
        "trials": [],
    }
    failures: list[str] = []
    try:
        for name in args.cases:
            for seed in args.seeds:
                print(f"Training {name}, seed={seed}, steps={steps}", flush=True)
                trial = run_trial(
                    CASES[name],
                    output / f"{name}-{seed}",
                    args.device,
                    steps,
                    seed,
                    args.profile == "smoke",
                )
                report["trials"].append(trial)
                failures.extend(
                    f"{name}/{seed}: {message}" for message in trial["quality_failures"]
                )
                print(f"{name}/{seed}: {json.dumps(trial['metrics'])}", flush=True)
        if args.baseline:
            failures.extend(compare(report, args.baseline))
    except Exception as error:
        failures.append(f"Execution failed: {error}")
    report.update(
        status="failed" if failures else ("smoke_passed" if args.profile == "smoke" else "passed"),
        failures=failures,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['status']}; report: {output / 'report.json'}", flush=True)
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
