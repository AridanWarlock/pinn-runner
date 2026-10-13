"""Verify benchmark mathematics and error gates without long neural training."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.execution import fingerprint
from benchmarks.metrics import quality_failures, regression_failures, solution_errors
from pinnstorch.api import Coordinates, Equation
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings
from scripts.benchmark_solutions import compare


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
    if name == "heat2d":
        field = (
            torch.sin(x * torch.pi)
            * torch.sin(coordinates.y * torch.pi)
            * torch.exp(-0.2 * torch.pi**2 * time)
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
