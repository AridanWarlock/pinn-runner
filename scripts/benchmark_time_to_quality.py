"""Short paired CUDA screening; analytical evaluation follows completed training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.cases import CASES  # noqa: E402
from benchmarks.metrics import solution_errors  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402
from pinnstorch.training.refinement import FixedObjective, LBFGSRefinement  # noqa: E402
from scripts.benchmark_solutions import environment  # noqa: E402

SNAPSHOTS = (0, 25, 50, 100, 150, 200, 300)
LIMITS = {"relative_l2": 0.001, "final_relative_l2": 0.002, "pde_rmse": 0.005}


def crossing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report an observed interval, never interpolate an unobserved crossing."""
    for index in range(len(rows) - 1):
        if rows[index]["passed"] and rows[index + 1]["passed"]:
            return {
                "status": "observed",
                "lower_seconds": rows[index - 1]["seconds"] if index else None,
                "upper_seconds": rows[index]["seconds"],
                "confirmed_seconds": rows[index + 1]["seconds"],
                "left_censored": index == 0,
            }
    return {"status": "not_confirmed", "observed_until_seconds": rows[-1]["seconds"]}


def trial(directory: Path, backend: str, seed: int) -> dict[str, Any]:
    case = CASES["heat3d"]
    directory.mkdir()
    config = case.config("cuda", 3000, seed)
    config["training"].update(constraints="hard_initial_dirichlet", hard_execution=backend)
    config["training"]["refinement"] = {"steps": 300}
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    settings = TaskSettings.load(path)
    problem = load_problem(case.source)
    factory = SolverFactory(settings, problem, directory, case.source.parent)
    objects = factory.build(predicting=False)
    torch.cuda.synchronize()
    started = time.perf_counter()
    objects.trainer.fit(objects.model, datamodule=objects.data)
    objects.model.to("cuda")
    refinement = settings.training.refinement
    stage = LBFGSRefinement(
        objects.model,
        FixedObjective.capture(objects.data, objects.model.device),
        refinement.learning_rate,
        refinement.history_size,
        refinement.tolerance_grad,
        refinement.tolerance_change,
    )
    rows: list[dict[str, Any]] = []
    overhead = 0.0
    for step in SNAPSHOTS:
        stage.run(step - stage.steps)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        snapshot_started = time.perf_counter()
        torch.save(objects.model.state_dict(), directory / f"lb-{step}.pt")
        overhead += time.perf_counter() - snapshot_started
        rows.append({"lb_steps": step, "seconds": elapsed, "closures": stage.closures})
    training_seconds = time.perf_counter() - started
    # No analytical value is inspected until all optimizer calls are complete.
    evaluation_config = case.config("cuda", 3000, seed, evaluating=True)
    evaluation_config["training"].update(
        constraints="hard_initial_dirichlet", hard_execution="eager"
    )
    path.write_text(yaml.safe_dump(evaluation_config), encoding="utf-8")
    evaluator = SolverFactory(TaskSettings.load(path), problem, directory, case.source.parent)
    evaluation = evaluator.build(predicting=True)
    model = evaluation.model.to("cuda").eval()
    spatial, clock, _ = evaluation.mesh.flatten_mesh(None)
    features = [
        torch.tensor(spatial[:, i : i + 1], dtype=torch.float32, device="cuda") for i in range(3)
    ]
    times = torch.tensor(clock, dtype=torch.float32, device="cuda")
    axes = {
        name: evaluation.mesh.spatial_domain_mesh[:, 0, i : i + 1].astype(np.float64)
        for i, name in enumerate(("x", "y", "z"))
    }
    grid_time = np.asarray(evaluation.mesh.time_domain, dtype=np.float64).reshape(-1, 1)
    points = np.random.default_rng(20261013).uniform(0, 1, (2048, 4))
    points[:, -1] *= 0.5
    for row in rows:
        model.load_state_dict(
            torch.load(directory / f"lb-{row['lb_steps']}.pt", map_location="cuda")
        )
        with torch.no_grad():
            chunks = [
                model([x[start : start + 2048] for x in features], times[start : start + 2048])[
                    "u"
                ]
                for start in range(0, len(times), 2048)
            ]
            values = torch.cat(chunks).cpu().numpy().astype(np.float64)
        metrics = solution_errors(case, axes, grid_time, values.reshape(len(axes["x"]), -1))
        columns = [
            torch.tensor(
                points[:, i : i + 1], dtype=torch.float32, device="cuda", requires_grad=True
            )
            for i in range(4)
        ]
        physical = dict(model(columns[:-1], columns[-1]))
        residuals = evaluator.adapter.residuals(physical, *columns)
        residual = torch.cat([residuals[name] for name in problem.equations], dim=1)
        metrics["pde_rmse"] = float(residual.detach().square().mean().sqrt().cpu())
        row.update(
            metrics=metrics, passed=all(metrics[name] <= limit for name, limit in LIMITS.items())
        )
    return {
        "backend": backend,
        "seed": seed,
        "snapshots": rows,
        "crossing": crossing(rows),
        "snapshot_seconds": overhead,
        "training_seconds_with_snapshots": training_seconds,
        "adam_steps": objects.trainer.global_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires the remote CUDA machine")
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"environment": environment("cuda"), "limits": LIMITS, "trials": []}
    for seed, backends in ((1234, ("eager", "product_rule")), (19, ("product_rule", "eager"))):
        for backend in backends:
            result = trial(args.output / f"{backend}-{seed}", backend, seed)
            report["trials"].append(result)
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(
                json.dumps({"backend": backend, "seed": seed, "crossing": result["crossing"]}),
                flush=True,
            )


if __name__ == "__main__":
    main()
