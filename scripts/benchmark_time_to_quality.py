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
from benchmarks.convergence import (  # noqa: E402
    ConvergenceArm,
    ConvergencePlan,
    convergence_trial,
    crossing,
)
from pinnstorch.runner.settings import TaskSettings  # noqa: E402
from scripts.benchmark_solutions import environment  # noqa: E402

__all__ = ["crossing"]
SNAPSHOTS = (0, 25, 50, 100, 150, 200, 300)
LIMITS = {"relative_l2": 0.001, "final_relative_l2": 0.002, "pde_rmse": 0.005}


def arguments() -> tuple[argparse.Namespace, ConvergencePlan, list[ConvergenceArm]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=tuple(name for name in CASES if name.startswith("heat")),
        default="heat3d",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 19])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--snapshots", type=int, nargs="+", default=list(SNAPSHOTS))
    parser.add_argument("--points", type=int)
    parser.add_argument(
        "--constraints",
        choices=("soft", "hard_initial_dirichlet", "hard_initial_robin"),
        default="hard_initial_dirichlet",
    )
    parser.add_argument(
        "--hard-execution",
        choices=("eager", "product_rule", "robin_batch", "robin_grouped"),
        nargs="+",
        default=["eager", "product_rule"],
    )
    parser.add_argument(
        "--hard-lifting", choices=("recursive", "expanded"), nargs="+", default=["recursive"]
    )
    parser.add_argument(
        "--backward", choices=("auto", "all", "parameters"), nargs="+", default=["all"]
    )
    parser.add_argument("--relative-l2", type=float, default=LIMITS["relative_l2"])
    parser.add_argument("--final-relative-l2", type=float, default=LIMITS["final_relative_l2"])
    parser.add_argument("--pde-rmse", type=float, default=LIMITS["pde_rmse"])
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Unique nonnegative seeds required")
    if any(
        len(set(values)) != len(values)
        for values in (args.hard_execution, args.hard_lifting, args.backward)
    ):
        parser.error("Duplicate experiment arms are not allowed")
    limits = {
        "relative_l2": args.relative_l2,
        "final_relative_l2": args.final_relative_l2,
        "pde_rmse": args.pde_rmse,
    }
    plan = ConvergencePlan(
        CASES[args.case], args.steps, tuple(args.snapshots), limits, args.constraints, args.points
    )
    arms = [
        ConvergenceArm(*values)
        for values in itertools.product(args.hard_execution, args.hard_lifting, args.backward)
    ]
    # Validate combinations before creating files or allocating CUDA models.
    for arm in arms:
        TaskSettings.read(plan.config(args.seeds[0], arm))
    return args, plan, arms


def main() -> None:
    args, plan, arms = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires the remote CUDA machine")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "contract": 2,
        "environment": environment("cuda"),
        "case": plan.case.name,
        "limits": plan.limits,
        "adam_steps": plan.steps,
        "lbfgs_snapshots": plan.snapshots,
        "constraints": plan.constraints,
        "points_override": plan.points,
        "limitations": "Only L-BFGS snapshots after completed Adam. Two adjacent successes required. Snapshot I/O overhead included in wall times and reported separately; no interpolation. Analytical evaluation follows all optimizer steps within each trial, never changes training or stopping. Physical autograd used independently of derivative executor. Final warm inference is measured separately.",
        "trials": [],
    }
    for index, seed in enumerate(args.seeds):
        for arm in arms if index % 2 == 0 else list(reversed(arms)):
            gc.collect()
            torch.cuda.empty_cache()
            result = convergence_trial(plan, arm, seed, args.output / f"{arm.name}-{seed}")
            report["trials"].append(result)
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(
                json.dumps({"arm": arm.name, "seed": seed, "crossing": result["crossing"]}),
                flush=True,
            )


if __name__ == "__main__":
    main()
