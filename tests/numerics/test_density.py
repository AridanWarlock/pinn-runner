"""Real CPU optimizer and sampling checks; these are not CUDA measurements."""

from pathlib import Path

import numpy as np
import torch

from benchmarks.cases import CASES
from benchmarks.convergence import ConvergenceArm, ConvergencePlan
from benchmarks.density import density_training, nested_points
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings


def test_nested_samples_preserve_bounds_prefix_and_global_rng() -> None:
    np.random.seed(19)
    before = np.random.get_state()
    low, high = [-2.0, 3.0, 0.4], [4.0, 5.0, 2.4]
    small = nested_points(19, 8, low, high)
    large = nested_points(19, 32, low, high)
    np.testing.assert_array_equal(small, large[:8])
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])
    assert np.all(large >= low) and np.all(large <= high)
    assert not np.array_equal(small, nested_points(73, 8, low, high))


def test_density_budget_uses_real_optimizer_and_identical_initial_weights(tmp_path: Path) -> None:
    plan = ConvergencePlan(
        CASES["heat_neumann_homogeneous2d"],
        2,
        (0, 100000),
        dict(relative_l2=1.0, final_relative_l2=1.0, pde_rmse=1.0),
        "hard_initial_robin",
        device="cpu",
        tolerance_change=1e-12,
        density=dict(
            training_seconds=0.4, snapshot_seconds=0.1, sampling="nested_uniform", prefix_points=8
        ),
    )
    results = []
    for count in (8, 32):
        arm = ConvergenceArm(
            "neumann_analytic", "recursive", "parameters", "auto", "neumann_polynomial", count
        )
        directory = tmp_path / str(count)
        directory.mkdir()
        factory = SolverFactory(
            TaskSettings.read(plan.config(19, arm)),
            load_problem(plan.case.source),
            directory,
            plan.case.source.parent,
        )
        objects = factory.build(predicting=False)
        rows, result = density_training(plan, arm, 19, objects, directory)
        results.append(result)
        assert result["training_seconds_with_snapshots"] >= 0.4
        assert result["stop_reason"] == "time_budget"
        assert 0 < result["adam_steps"] <= 2
        assert rows[0]["lb_steps"] == 0
        assert all(a["lb_steps"] < b["lb_steps"] for a, b in zip(rows, rows[1:]))
        assert all((directory / f"lb-{row['lb_steps']}.pt").is_file() for row in rows)
        assert all(torch.isfinite(p).all() for p in objects.model.parameters())
        assert len(objects.data.train_datasets[0]) == count
    assert results[0]["initial_state_sha256"] == results[1]["initial_state_sha256"]
    assert results[0]["fixed_objective_sha256"] == results[1]["objective_prefix_sha256"]
    assert results[0]["fixed_objective_sha256"] != results[1]["fixed_objective_sha256"]
