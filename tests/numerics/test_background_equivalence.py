"""The diagnostic replays actual training loss on identical data and weights."""

from pathlib import Path

import pytest
import torch

from benchmarks.background_equivalence import BackgroundEquivalence
from benchmarks.cases import CASES
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings


def test_background_probe_reconstructs_same_points_and_checks_real_gradients(
    tmp_path: Path,
) -> None:
    case = CASES["heat_neumann_polynomial2d"]
    config = case.config("cpu", 2, 19)
    config["network"]["hidden_layers"] = [8, 8]
    config["training"].update(
        constraints="hard_initial_robin", collocation={"sampling": "lhs", "points": 7}
    )
    probe = BackgroundEquivalence(
        TaskSettings.read(config), load_problem(case.source), tmp_path, case.source.parent
    )
    assert len(set(probe.collocation_hashes.values())) == 1
    assert probe.points is not None and probe.points.shape == (7, 3)
    result = probe.compare(7, torch.float64)
    assert result["loss_absolute_difference"] < 1e-12
    assert result["field_max_absolute_difference"] < 1e-12
    assert result["residual_max_absolute_difference"] < 1e-12
    assert result["gradient_relative_l2_difference"] < 1e-12
    assert result["general_self_repeat_exact"] and result["static_self_repeat_exact"]
    # Changing evaluation dtype must preserve the replay's original weights.
    result = probe.compare(7, torch.float32)
    assert result["residual_max_absolute_difference"] < 1e-5
    assert result["gradient_relative_l2_difference"] < 1e-5
    with pytest.raises(ValueError, match="fit the reconstructed"):
        probe.compare(8, torch.float64)


def test_background_probe_uses_declared_equation_and_field_names(tmp_path: Path) -> None:
    from dataclasses import replace

    from pinnstorch.api import Coordinates, Equation, TensorFields

    case = CASES["heat_neumann_polynomial1d"]
    original = load_problem(case.source)

    def initial(c: Coordinates) -> TensorFields:
        return {"temperature": original.initial(c)["u"]}

    def diffusion(e: Equation) -> torch.Tensor:
        return e.dt("temperature") - 0.1 * e.laplacian("temperature")

    problem = replace(
        original, fields=("temperature",), initial=initial, equations={"diffusion": diffusion}
    )
    config = case.config("cpu", 2, 19)
    config["network"]["hidden_layers"] = [4, 4]
    config["boundaries"][0]["field"] = "temperature"
    config["training"].update(
        constraints="hard_initial_robin", collocation={"sampling": "lhs", "points": 3}
    )
    probe = BackgroundEquivalence(TaskSettings.read(config), problem, tmp_path, case.source.parent)
    assert probe.compare(3, torch.float64)["gradient_relative_l2_difference"] < 1e-12
