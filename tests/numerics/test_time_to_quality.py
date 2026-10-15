from pathlib import Path
from typing import Any

import pytest

from scripts.benchmark_time_to_quality import crossing


def test_crossing_requires_two_adjacent_successes() -> None:
    rows = [
        {"seconds": 1.0, "passed": False},
        {"seconds": 2.0, "passed": True},
        {"seconds": 3.0, "passed": False},
        {"seconds": 4.0, "passed": True},
        {"seconds": 5.0, "passed": True},
    ]
    assert crossing(rows) == {
        "status": "observed",
        "lower_seconds": 3.0,
        "upper_seconds": 4.0,
        "confirmed_seconds": 5.0,
        "left_censored": False,
    }
    assert crossing(rows[:4])["status"] == "not_confirmed"


def test_crossing_marks_unknown_time_before_first_snapshot() -> None:
    result = crossing([{"seconds": 8.0, "passed": True}, {"seconds": 9.0, "passed": True}])
    assert result["left_censored"] is True
    assert result["lower_seconds"] is None


def test_crossing_requires_observations() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        crossing([])


def test_generic_robin_snapshots_preserve_training_and_evaluate_afterward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.cases import CASES, Array, BenchmarkCase
    from benchmarks.convergence import (
        ConvergenceArm,
        ConvergencePlan,
        SnapshotTraining,
        convergence_trial,
    )
    from benchmarks.variants import BenchmarkVariant

    class SmallGrid:
        def apply(self, config: dict[str, Any]) -> None:
            config["network"]["hidden_layers"] = [4]
            for axis in config["domain"]["space"].values():
                axis["points"] = 3
            config["domain"]["time"]["points"] = 3

    case = BenchmarkVariant("heat_neumann2d", 2, base=CASES["heat_neumann2d"], adapter=SmallGrid())
    plan = ConvergencePlan(case, 2, (0, 1, 2), {"pde_rmse": 10.0}, "hard_initial_robin", 7, "cpu")
    completed = False
    original_run = SnapshotTraining.run
    original_reference = type(case.base).reference

    def guarded_run(self: SnapshotTraining) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        nonlocal completed
        result = original_run(self)
        completed = True
        return result

    def guarded_reference(self: BenchmarkCase, spatial: dict[str, Array], time: Array) -> Array:
        assert completed, "Analytical reference accessed before training completed"
        return original_reference(self, spatial, time)

    monkeypatch.setattr(SnapshotTraining, "run", guarded_run)
    monkeypatch.setattr(type(case.base), "reference", guarded_reference)
    results = []
    for mode in ("all", "parameters"):
        completed = False
        results.append(
            convergence_trial(
                plan, ConvergenceArm("robin_grouped", "expanded", mode), 19, tmp_path / mode
            )
        )
    for result in results:
        assert result["adam_steps"] == 2
        assert result["prediction_points"] == 27
        assert result["snapshot_seconds"] >= 0
        rows = result["snapshots"]
        assert [r["lb_steps"] for r in rows] == [0, 1, 2]
        assert rows[0]["closures"] == 0
        assert rows[-1]["closures"] >= 2
        assert result["crossing"]["status"] == "observed"
        assert result["crossing"]["left_censored"]
    for a, b in zip(results[0]["snapshots"], results[1]["snapshots"]):
        assert a["metrics"] == b["metrics"]
        assert a["closures"] == b["closures"]
        assert a["lbfgs_residual_points"] == b["lbfgs_residual_points"]


def test_convergence_plan_rejects_invalid_budgets() -> None:
    from benchmarks.cases import CASES
    from benchmarks.convergence import ConvergencePlan

    for steps, snapshots, limits in (
        (0, (0, 1), {"pde_rmse": 1}),
        (2, (), {"pde_rmse": 1}),
        (2, (1, 2), {"pde_rmse": 1}),
        (2, (0, 2, 1), {"pde_rmse": 1}),
        (2, (0, 1), {"pde_rmse": float("nan")}),
        (2, (0, 1), {"pde_rmse": 0}),
    ):
        with pytest.raises(ValueError):
            ConvergencePlan(CASES["heat3d"], steps, snapshots, limits)
