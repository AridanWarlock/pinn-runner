"""Verify benchmark mathematics and error gates without long neural training."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.execution import fingerprint
from benchmarks.metrics import quality_failures, regression_failures, solution_errors
from benchmarks.policies import CartesianGrid, FixedQuality, TrainingBudget
from benchmarks.variants import BenchmarkVariant, BoundaryWeight
from pinnstorch.api import Coordinates, Equation
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings
from scripts.benchmark_solutions import compare


def test_variant_preserves_custom_grid_budget_and_acceptance() -> None:
    case = replace(
        CASES["heat2d"],
        grid=CartesianGrid(space_points=(9, 21)),
        budget=TrainingBudget(layers=(16, 16), learning_rate=0.002),
        quality=FixedQuality(relative_l2=0.02),
    )
    variant = BenchmarkVariant(case.name, case.dimensions, base=case, adapter=BoundaryWeight(3))
    config = variant.config("cpu", 100, 19)
    assert config["domain"]["space"]["x"]["points"] == 9
    assert config["network"]["hidden_layers"] == [16, 16]
    assert config["training"]["optimizer"]["learning_rate"] == 0.002
    metrics = dict(relative_l2=0.03, max_error=0, pde_rmse=0, initial_rmse=0, boundary_rmse=0)
    assert len(quality_failures(variant, metrics)) == 1
    assert not quality_failures(CASES["heat2d"], metrics)


@pytest.mark.parametrize("name", list(CASES))
def test_analytical_field_satisfies_equation_initial_and_boundaries(name: str) -> None:
    case = CASES[name]
    generator = torch.Generator().manual_seed(19)
    columns = [
        torch.rand((17, 1), dtype=torch.float64, generator=generator).requires_grad_()
        for _ in range(case.dimensions + 1)
    ]
    coordinates = Coordinates(dict(zip(("x", "y", "z"), columns[:-1])), columns[-1])
    x, time = coordinates.x, coordinates.t
    if name in ("heat2d", "heat3d"):
        field = (
            torch.sin(x * torch.pi)
            * torch.sin(coordinates.y * torch.pi)
            * torch.exp(-0.2 * torch.pi**2 * time)
        )
        if name == "heat3d":
            field = (
                field * torch.sin(coordinates.z * torch.pi) * torch.exp(-0.1 * torch.pi**2 * time)
            )
    elif name == "advection":
        field = torch.sin((x - time) * torch.pi)
    else:
        field = torch.exp(-time) * torch.sin(x * torch.pi)
    problem = load_problem(case.source)
    equation = Equation({"u": field}, coordinates)
    for function in problem.equations.values():
        torch.testing.assert_close(function(equation), torch.zeros_like(field), atol=2e-12, rtol=0)
    numpy_spatial = {axis: value.detach().numpy() for axis, value in coordinates.spatial.items()}
    np.testing.assert_allclose(
        field.detach().numpy(), case.reference(numpy_spatial, time.detach().numpy()), atol=1e-12
    )
    initial_time = torch.zeros_like(time)
    initial = problem.initial(Coordinates(coordinates.spatial, initial_time))["u"]
    np.testing.assert_allclose(
        initial.detach().numpy(), case.reference(numpy_spatial, initial_time.numpy()), atol=1e-12
    )
    settings = TaskSettings.read(case.config("cpu", 3, 1234))
    for boundary in settings.boundaries:
        boundary_spatial = {
            **numpy_spatial,
            boundary.axis: np.zeros_like(numpy_spatial[boundary.axis]),
        }
        reference = case.reference(boundary_spatial, time.detach().numpy())
        if boundary.value_function:
            observed = (
                problem.boundary_values[boundary.value_function](coordinates).detach().numpy()
            )
            np.testing.assert_allclose(observed, reference, atol=1e-12)
        else:
            np.testing.assert_allclose(reference, 0, atol=1e-12)
            boundary_spatial[boundary.axis] = np.ones_like(boundary_spatial[boundary.axis])
            np.testing.assert_allclose(
                case.reference(boundary_spatial, time.detach().numpy()), 0, atol=1e-12
            )


@pytest.mark.parametrize("name", list(CASES))
def test_error_metrics_detect_wrong_solution(name: str) -> None:
    case = CASES[name]
    settings = TaskSettings.read(case.config("cpu", 3, 1234, evaluating=True))
    grid = settings.domain.grid()
    spatial = {axis: grid.spatial[i] for i, axis in enumerate(settings.domain.space)}
    # PointCloudData contains Cartesian spatial rows, independent of time.
    expected = case.reference(spatial, grid.time.T)
    exact = solution_errors(case, spatial, grid.time, expected)
    assert exact["relative_l2"] == 0
    bad = solution_errors(case, spatial, grid.time, np.zeros_like(expected))
    assert bad["relative_l2"] == pytest.approx(1)
    assert quality_failures(case, {**bad, "pde_rmse": 0})
    with pytest.raises(ValueError, match="finite"):
        solution_errors(case, spatial, grid.time, np.full_like(expected, np.nan))


def test_regression_gate_and_task_identity(tmp_path: Path) -> None:
    metrics = {"relative_l2": 0.01, "max_error": 0.03, "pde_rmse": 0.02}
    assert not regression_failures(metrics, metrics)
    assert regression_failures({**metrics, "relative_l2": 0.1}, metrics)
    assert not regression_failures({key: 1e-5 for key in metrics}, {key: 0 for key in metrics})
    case = CASES["heat2d"]
    identity = fingerprint(case, "cuda", 3000, 1234)
    assert identity != fingerprint(case, "cuda", 3001, 1234)
    assert identity != fingerprint(case, "cuda", 3000, 4321)
    report = {
        "contract": 1,
        "profile": "standard",
        "status": "passed",
        "trials": [{"case": case.name, "seed": 1234, "fingerprint": identity, "metrics": metrics}],
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(report))
    assert not compare(report, baseline)
    changed = {**report, "trials": [{**report["trials"][0], "fingerprint": "different"}]}
    with pytest.raises(ValueError, match="differs"):
        compare(changed, baseline)


@pytest.mark.e2e
def test_benchmark_smoke_exercises_public_runner(tmp_path: Path) -> None:
    output = tmp_path / "report"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_solutions.py",
            "--device",
            "cpu",
            "--profile",
            "smoke",
            "--seeds",
            "1234",
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "smoke_passed"
    assert len(report["trials"]) == 3
    for trial in report["trials"]:
        assert trial["steps"] == 3
        assert trial["quality_checked"] is False
        assert all(np.isfinite(value) for value in trial["metrics"].values())
        performance = trial["performance"]
        assert performance["training_steps_per_second"] == pytest.approx(
            3 / trial["training_seconds"]
        )
        assert performance["prediction_points_per_second"] > 0
        assert performance["warm_prediction_seconds"] > 0
    reused = tmp_path / "reused"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_solutions.py",
            "--device",
            "cpu",
            "--profile",
            "smoke",
            "--seeds",
            "1234",
            "--cases",
            "advection",
            "--output-dir",
            str(reused),
            "--measure-existing",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    measured = json.loads((reused / "report.json").read_text())["trials"][0]
    original = next(trial for trial in report["trials"] if trial["case"] == "advection")
    assert measured["training_seconds"] == original["training_seconds"]
    assert not (reused / "advection-1234/training.log").exists()
    assert measured["metrics"] == original["metrics"]


def test_nonfinite_baseline_is_rejected() -> None:
    metrics = {"relative_l2": 0.01, "max_error": 0.03, "pde_rmse": 0.02}
    with pytest.raises(ValueError, match="finite"):
        regression_failures(metrics, {**metrics, "relative_l2": float("nan")})


def test_incompatible_baseline_stops_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.benchmark_solutions as command

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "contract": 1,
                "profile": "standard",
                "status": "passed",
                "trials": [{"case": "heat2d", "seed": 1234, "fingerprint": "wrong"}],
            }
        )
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--cases",
            "heat2d",
            "--seeds",
            "1234",
            "--output-dir",
            str(output),
            "--baseline",
            str(baseline),
        ],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(command, "environment", lambda device: {"device": device})

    def unexpected_training(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid baseline must not spend GPU time")

    monkeypatch.setattr(command, "run_trial", unexpected_training)
    assert command.main() == 1
    report = json.loads((output / "report.json").read_text())
    assert report["trials"] == []
    assert "Baseline task/budget/evaluation differs" in report["failures"][0]


def test_summary_keeps_seed_spread() -> None:
    from scripts.benchmark_solutions import summarize

    summary = summarize(
        [
            {
                "case": "heat2d",
                "seed": 1,
                "metrics": {"relative_l2": 0.01},
                "training_seconds": 10,
            },
            {
                "case": "heat2d",
                "seed": 2,
                "metrics": {"relative_l2": 0.03},
                "training_seconds": 12,
            },
        ]
    )["heat2d"]
    assert summary["relative_l2_median"] == pytest.approx(0.02)
    assert summary["relative_l2_max"] == 0.03
    assert summary["training_seconds_median"] == 11


def test_prediction_timing_handles_batches_on_cpu() -> None:
    from benchmarks.performance import prediction_speed
    from pinnstorch.contracts import SamplerBatch, TensorOutputs

    class ToyPredictor:
        points = 0

        def predict_step(self, batch: SamplerBatch, batch_idx: int) -> TensorOutputs:
            spatial, times, _ = batch
            self.points += len(times)
            return {"u": spatial[0] + times}

    model = ToyPredictor()
    clock = torch.zeros((5, 1))
    performance = prediction_speed(model, [clock.clone()], clock, batch_size=2)
    assert performance["prediction_points"] == 5
    assert model.points == 5 * (
        performance["prediction_warmup_passes"] + performance["prediction_measurement_passes"]
    )
    assert performance["warm_prediction_seconds"] > 0
    assert performance["warm_prediction_points_per_second"] == pytest.approx(
        5 / performance["warm_prediction_seconds"]
    )
    with pytest.raises(ValueError, match="aligned"):
        prediction_speed(model, [clock[:2]], clock, batch_size=2)


def test_temporal_diagnostics_distinguish_decay_from_absolute_error() -> None:
    from benchmarks.metrics import temporal_errors

    case = CASES["heat3d"]
    spatial = {axis: np.array([[0.0], [0.5], [1.0]]) for axis in ("x", "y", "z")}
    time = np.array([[0.0], [0.5]])
    exact = case.reference(spatial, time.T)
    rows = temporal_errors(case, spatial, time, exact + 0.01)
    assert rows[0]["rmse"] == pytest.approx(rows[1]["rmse"])
    assert rows[0]["boundary_rmse"] == pytest.approx(0.01)
    assert rows[0]["interior_rmse"] == pytest.approx(0.01)
    assert float(rows[1]["relative_l2"] or 0) > float(rows[0]["relative_l2"] or 0) * 4
    assert [row["time"] for row in rows] == [0, 0.5]


def test_temporal_diagnostics_handle_zero_reference_and_missing_boundary() -> None:
    from benchmarks.metrics import temporal_errors

    case = CASES["advection"]
    spatial = {"x": np.array([[0.5]])}
    time = np.array([[0.5]])
    rows = temporal_errors(case, spatial, time, np.array([[0.1]]))
    assert rows[0]["relative_l2"] is None
    assert rows[0]["boundary_rmse"] is None
    assert rows[0]["interior_rmse"] == pytest.approx(0.1)
