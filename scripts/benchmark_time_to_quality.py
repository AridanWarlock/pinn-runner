"""Paired snapshot convergence; analytical evaluation happens after completed training."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.cases import CASES  # noqa: E402
from benchmarks.catalog import (  # noqa: E402
    BACKGROUNDS,
    BACKWARDS,
    CONSTRAINTS,
    CONVERGENCE_CASES,
    ENCODINGS,
    EXECUTIONS,
    LIFTINGS,
)
from benchmarks.convergence import (  # noqa: E402
    ConvergenceArm,
    ConvergencePlan,
    convergence_trial,
    crossing,
)
from benchmarks.convergence_processes import run_isolated, write_report  # noqa: E402
from pinnstorch.runner.settings import Refinement, TaskSettings  # noqa: E402
from scripts.benchmark_solutions import environment  # noqa: E402

__all__ = ["crossing"]
SNAPSHOTS = (0, 25, 50, 100, 150, 200, 300)
LIMITS = {"relative_l2": 0.001, "final_relative_l2": 0.002, "pde_rmse": 0.005}


def arguments() -> tuple[argparse.Namespace, ConvergencePlan, list[ConvergenceArm]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shared-process",
        action="store_true",
        help="Diagnostic only: reuse this process for all trials instead of isolating each trial",
    )
    parser.add_argument(
        "--case",
        choices=CONVERGENCE_CASES,
        default="heat3d",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 19])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--snapshots", type=int, nargs="+", default=list(SNAPSHOTS))
    parser.add_argument("--points", type=int)
    parser.add_argument("--density", type=json.loads)
    parser.add_argument(
        "--constraints",
        choices=CONSTRAINTS,
        default="hard_initial_dirichlet",
    )
    parser.add_argument(
        "--hard-execution",
        choices=EXECUTIONS,
        nargs="+",
        default=["eager", "product_rule"],
    )
    parser.add_argument("--hard-lifting", choices=LIFTINGS, nargs="+", default=["recursive"])
    parser.add_argument("--backward", choices=BACKWARDS, nargs="+", default=["all"])
    parser.add_argument("--hard-background", choices=BACKGROUNDS, nargs="+", default=["general"])
    parser.add_argument(
        "--spatial-encoding",
        choices=ENCODINGS,
        nargs="+",
        default=["identity"],
        help="Neumann features use eager or analytic derivatives without projection",
    )
    parser.add_argument("--relative-l2", type=float, default=LIMITS["relative_l2"])
    parser.add_argument("--final-relative-l2", type=float, default=LIMITS["final_relative_l2"])
    parser.add_argument("--pde-rmse", type=float, default=LIMITS["pde_rmse"])
    parser.add_argument("--mode-relative-l2", type=float, default=None)
    parser.add_argument(
        "--lbfgs-tolerance-change", type=float, default=Refinement().tolerance_change
    )
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Unique nonnegative seeds required")
    if any(
        len(set(values)) != len(values)
        for values in (
            args.hard_execution,
            args.hard_lifting,
            args.backward,
            args.hard_background,
            args.spatial_encoding,
        )
    ):
        parser.error("Duplicate experiment arms are not allowed")
    limits = {
        "relative_l2": args.relative_l2,
        "final_relative_l2": args.final_relative_l2,
        "pde_rmse": args.pde_rmse,
    }
    if args.mode_relative_l2 is not None:
        limits["mode_relative_l2"] = args.mode_relative_l2
    plan = ConvergencePlan(
        CASES[args.case],
        args.steps,
        tuple(args.snapshots),
        limits,
        args.constraints,
        args.points,
        tolerance_change=args.lbfgs_tolerance_change,
        density=args.density,
    )
    arms = list(
        dict.fromkeys(
            ConvergenceArm(
                execution
                if encoding == "identity" or execution == "neumann_analytic"
                else "eager",
                lifting,
                backward,
                background,
                encoding,
                args.points if args.density is not None else None,
            )
            for execution, lifting, backward, background, encoding in itertools.product(
                args.hard_execution,
                args.hard_lifting,
                args.backward,
                args.hard_background,
                args.spatial_encoding,
            )
        )
    )
    # Validate combinations before creating files or allocating CUDA models.
    for arm in arms:
        TaskSettings.read(plan.config(args.seeds[0], arm))
    return args, plan, arms


def main() -> None:
    args, plan, arms = arguments()
    if not args.shared_process:
        run_isolated(plan, arms, args.seeds, args.output.resolve(), Path(__file__).resolve())
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires the remote CUDA machine")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        **({"density": plan.density} if plan.density is not None else {}),
        "contract": 3,
        "status": "running",
        "execution_lifecycle": "single_trial"
        if len(args.seeds) * len(arms) == 1
        else "shared_process",
        "environment": environment("cuda"),
        "case": plan.case.name,
        "limits": plan.limits,
        "adam_steps": plan.steps,
        "lbfgs_snapshots": plan.snapshots,
        "constraints": plan.constraints,
        "points_override": plan.points,
        "lbfgs_tolerance_change": plan.tolerance_change,
        "limitations": "Only L-BFGS snapshots after completed Adam. Two adjacent successes required; no interpolation. Previous snapshot I/O included and reported separately. Analytical evaluation follows all optimizer steps, never changes training. Quality uses independent eager physical autograd. Prediction timing of the trained executor and eager control are separate. Shared-process reuse is diagnostic and may affect FP32 repeatability.",
        "trials": [],
    }
    for index, seed in enumerate(args.seeds):
        for arm in arms if index % 2 == 0 else list(reversed(arms)):
            gc.collect()
            torch.cuda.empty_cache()
            result = convergence_trial(plan, arm, seed, args.output / f"{arm.name}-{seed}")
            report["trials"].append(result)
            write_report(args.output, report)
            print(
                json.dumps({"arm": arm.name, "seed": seed, "crossing": result["crossing"]}),
                flush=True,
            )
    report["status"] = "completed"
    write_report(args.output, report)


if __name__ == "__main__":
    main()
