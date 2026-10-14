"""Check physical boundary operators and integral heat balance of trained 1D cases."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.heat_balance import heat_balance  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402

LIMITS = {
    "relative_l2": 0.01,
    "final_relative_l2": 0.02,
    "pde_rmse": 0.02,
    "boundary_operator_rmse": 1e-4,
    "heat_balance_rate_rmse": 0.01,
    "heat_balance_accumulated_max_error": 0.005,
    "initial_rmse": 1e-5,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This numerical benchmark requires remote CUDA")
    if args.report.exists():
        raise ValueError("Report already exists")
    result = json.loads((args.trial / "result.json").read_text())
    name = result["case"]
    if name not in ("heat_neumann", "heat_robin"):
        raise ValueError("Balance benchmark only supports the documented interval cases")
    task = args.trial / "task"
    settings = TaskSettings.load(task / "config.yaml")
    problem = load_problem(task / "functions.py")
    factory = SolverFactory(settings, problem, args.trial, task)
    model = factory.build(predicting=True).model.to("cuda").eval()
    model.load_state_dict(
        torch.load(args.trial / "training/checkpoints/final.ckpt", map_location="cpu")[
            "state_dict"
        ],
        strict=True,
    )

    def field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model.forward([x], t)["u"]

    def target(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return factory.adapter.boundary_value("wall", [x], t)

    def source(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x

    metrics = heat_balance(
        field,
        settings.domain.space["x"].interval,
        settings.domain.time.interval,
        "cuda",
        math.pi / 2 if name == "heat_robin" else 0,
        1,
        target,
        source,
    )
    combined = {**result["metrics"], **metrics}
    failures = [
        f"{key}={combined[key]:.6g} exceeds {limit}"
        for key, limit in LIMITS.items()
        if not math.isfinite(combined[key]) or combined[key] > limit
    ]
    report = {
        "case": name,
        "seed": result["seed"],
        "constraint_kind": result["constraint_kind"],
        "limits": LIMITS,
        "metrics": combined,
        "status": "failed" if failures else "passed",
        "failures": failures,
        "space_points": 257,
        "time_points": 31,
        "conductivity": 0.1,
        "source": "x",
        "limitations": "Quadrature error included; physical autograd on a grid independent of training points",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
