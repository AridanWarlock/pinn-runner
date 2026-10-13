"""Temporal holdout correctness and checkpoint evaluation without optimization."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from benchmarks.cases import CASES
from benchmarks.diagnostics import residual_time_bins, worst_time_error
from benchmarks.execution import evaluate_checkpoint
from pinnstorch.models import PINNModule
from pinnstorch.runner.factory import ExecutionObjects, SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings


def test_residual_bins_cover_endpoints_without_double_counting() -> None:
    times = np.array([[0], [0.25], [0.5]])
    residual = np.array([[1.0, 3], [2, 4], [5, 6]])
    rows = residual_time_bins(times, residual, (0, 0.5), 2)
    assert [row["points"] for row in rows] == [1, 2]
    assert rows[0]["rmse"] == pytest.approx(np.sqrt(5))
    assert rows[1]["rmse"] == pytest.approx(np.sqrt(81 / 4))
    assert rows[1]["max_absolute"] == 6
    with pytest.raises(ValueError, match="inside"):
        residual_time_bins(times + 1, residual, (0, 0.5))


def test_empty_bins_and_zero_reference_are_explicit() -> None:
    rows = residual_time_bins(np.array([[0.0]]), np.array([[0.0]]), (0, 0.5), 2)
    assert rows[0]["rmse"] == 0
    assert rows[1]["rmse"] is None
    assert worst_time_error([{"time": 0, "relative_l2": None}]) == {
        "worst_time_l2": None,
        "worst_time": None,
        "undefined_layers": 1,
    }
    result = worst_time_error(
        [
            {"time": 0, "relative_l2": 0.1},
            {"time": 0.5, "relative_l2": 0.4},
            {"time": 0.7, "relative_l2": None},
        ]
    )
    assert result["worst_time_l2"] == 0.4
    assert result["worst_time"] == 0.5
    assert result["undefined_layers"] == 1


def test_checkpoint_diagnostics_leave_weights_and_gradients_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = CASES["advection"]
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(case.config("cpu", 3, 19, True)))
    (tmp_path / "functions.py").write_bytes(case.source.read_bytes())
    settings = TaskSettings.load(tmp_path / "config.yaml")
    problem = load_problem(tmp_path / "functions.py")
    factory = SolverFactory(settings, problem, tmp_path / "initial", tmp_path)
    objects = factory.build(predicting=True)
    expected = {key: value.clone() for key, value in objects.model.state_dict().items()}
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": expected}, checkpoint)
    observed: list[PINNModule] = []
    build = SolverFactory.build

    def capture(self: SolverFactory, predicting: bool) -> ExecutionObjects:
        result = build(self, predicting)
        observed.append(result.model)
        return result

    monkeypatch.setattr(SolverFactory, "build", capture)
    first = evaluate_checkpoint(case, tmp_path, tmp_path / "first", checkpoint, "cpu")
    second = evaluate_checkpoint(case, tmp_path, tmp_path / "second", checkpoint, "cpu")
    assert first.residual_rmse == second.residual_rmse
    assert first.residual_bins == second.residual_bins
    assert sum(int(row["points"] or 0) for row in first.residual_bins) == 2048
    for model in observed:
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in model.parameters())
