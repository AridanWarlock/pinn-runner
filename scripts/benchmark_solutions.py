"""Numerical PDE acceptance: train, evaluate against analytics, compare a baseline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
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
from benchmarks.execution import fingerprint, measure_existing, run_trial  # noqa: E402
from benchmarks.metrics import regression_failures  # noqa: E402
from benchmarks.variants import (  # noqa: E402
    BenchmarkVariant,
    BoundaryWeight,
    CollocationRefresh,
    Constraints,
    HardExecution,
    RateDecay,
    RefinementSteps,
    TrainingComposition,
)
from pinnstorch.runner.settings import LossSettings, Reader, ScheduleSettings  # noqa: E402


def environment(device: str) -> dict[str, Any]:
    pinned = os.environ.get("PINN_PINNED_REVISION")
    if pinned:
        revision_value = pinned
        working_tree_dirty = False
    else:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        revision_value = revision.stdout.strip()
        working_tree_dirty = bool(dirty.stdout.strip())
    return {
        "revision": revision_value,
        "working_tree_dirty": working_tree_dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "packages": {name: version(name) for name in ("numpy", "lightning")},
    }


def validate_baseline(report: dict[str, Any], path: Path) -> dict[tuple[str, int], Any]:
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
    for key, trial in current.items():
        old = previous[key]
        if trial["fingerprint"] != old["fingerprint"]:
            raise ValueError(f"Baseline task/budget/evaluation differs for {key}")
        regression_failures(old["metrics"], old["metrics"])
    return previous


def compare(report: dict[str, Any], path: Path) -> list[str]:
    previous = validate_baseline(report, path)
    failures: list[str] = []
    for trial in report["trials"]:
        key = (trial["case"], trial["seed"])
        old = previous[key]
        failures.extend(
            f"{key}: {message}"
            for message in regression_failures(trial["metrics"], old["metrics"])
        )
    return failures


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in dict.fromkeys(trial["case"] for trial in trials):
        group = [trial for trial in trials if trial["case"] == name]
        errors = [trial["metrics"]["relative_l2"] for trial in group]
        performance = {
            key: statistics.median(trial["performance"][key] for trial in group)
            for key in group[0].get("performance", {})
        }
        summary[name] = {
            "seeds": [trial["seed"] for trial in group],
            "relative_l2_median": statistics.median(errors),
            "relative_l2_min": min(errors),
            "relative_l2_max": max(errors),
            "performance_median": performance,
            "training_seconds_median": statistics.median(
                trial["training_seconds"] for trial in group
            ),
        }
    return summary


def arguments() -> tuple[argparse.Namespace, int]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--profile", choices=("smoke", "standard", "reference"), default="standard"
    )
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(CASES), default=["heat2d", "advection", "burgers"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 4321])
    parser.add_argument("--steps", type=int, help="Override the fixed profile budget")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="A new directory for this run"
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--measure-existing",
        type=Path,
        help="Reevaluate saved checkpoints and add timings without training",
    )
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--gradient-norm-every", type=int, default=0)
    parser.add_argument("--lr-final-ratio", type=float, default=1.0)
    parser.add_argument("--resample-every-epochs", type=int, default=0)
    parser.add_argument("--collocation-points", type=int, help="Override the case point budget")
    parser.add_argument(
        "--lbfgs-steps", type=int, default=0, help="Additional full-objective steps"
    )
    parser.add_argument(
        "--constraints", choices=("soft", "hard_initial_dirichlet"), default="soft"
    )
    parser.add_argument(
        "--hard-execution", choices=("eager", "fused_predict", "product_rule"), default="eager"
    )
    args = parser.parse_args()
    if (
        args.lbfgs_steps < 0
        or args.resample_every_epochs < 0
        or (args.collocation_points is not None and args.collocation_points < 1)
    ):
        parser.error("Sampling counts must be positive; refresh interval nonnegative")
    ScheduleSettings.read(
        Reader({"kind": "exponential", "final_ratio": args.lr_final_ratio}, "experiment.schedule")
    )
    LossSettings.read(
        Reader(
            {
                "kind": "fixed",
                "boundaries": args.boundary_weight,
                "gradient_norm_every": args.gradient_norm_every,
            },
            "experiment.loss",
        )
    )
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
    return args, steps


def main() -> int:
    args, steps = arguments()
    cases = {
        name: BenchmarkVariant(
            case.name,
            case.dimensions,
            base=case,
            adapter=TrainingComposition(
                (
                    BoundaryWeight(args.boundary_weight, args.gradient_norm_every),
                    RateDecay(args.lr_final_ratio),
                    CollocationRefresh(args.resample_every_epochs, args.collocation_points),
                    Constraints(args.constraints),
                    HardExecution(args.hard_execution),
                    RefinementSteps(args.lbfgs_steps),
                )
            ),
        )
        for name, case in CASES.items()
    }
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
        if args.measure_existing:
            origin = json.loads(
                (args.measure_existing / "report.json").read_text(encoding="utf-8")
            )
            report["training_origin"] = {
                "directory": str(args.measure_existing.resolve()),
                "revision": origin["environment"]["revision"],
            }
        if args.baseline:
            planned = [
                {
                    "case": name,
                    "seed": seed,
                    "fingerprint": fingerprint(cases[name], args.device, steps, seed),
                }
                for name in args.cases
                for seed in args.seeds
            ]
            validate_baseline({**report, "trials": planned}, args.baseline)
        for name in args.cases:
            for seed in args.seeds:
                action = "Reevaluating" if args.measure_existing else "Training"
                print(f"{action} {name}, seed={seed}, steps={steps}", flush=True)
                if args.measure_existing:
                    trial = measure_existing(
                        cases[name],
                        args.measure_existing / f"{name}-{seed}",
                        output / f"{name}-{seed}",
                        args.device,
                        steps,
                        seed,
                        args.profile == "smoke",
                    )
                else:
                    trial = run_trial(
                        cases[name],
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
                print(f"Performance: {json.dumps(trial['performance'])}", flush=True)
        if args.baseline:
            failures.extend(compare(report, args.baseline))
    except Exception as error:
        failures.append(f"Execution failed: {error}")
    report.update(
        status="failed" if failures else ("smoke_passed" if args.profile == "smoke" else "passed"),
        failures=failures,
        summary=summarize(report["trials"]),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['status']}; report: {output / 'report.json'}", flush=True)
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
