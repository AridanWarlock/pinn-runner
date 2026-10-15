"""Automatic adoption is limited to tested constraints and runtime combinations."""

from typing import Any

import pytest

from benchmarks.cases import CASES
from pinnstorch.runner.backward import resolve_backward
from pinnstorch.runner.settings import TaskSettings


@pytest.mark.parametrize(
    "constraints,runtime,expected",
    [
        ("hard_initial_robin", {"device": "cpu"}, "parameters"),
        ("hard_initial_robin", {"device": "cuda", "devices": [0]}, "parameters"),
        ("hard_initial_robin", {"device": "mps"}, "all"),
        ("hard_initial_robin", {"device": "cuda", "devices": 2}, "all"),
        ("hard_initial_robin", {"device": "cuda", "precision": "16-mixed"}, "all"),
        ("soft", {"device": "cuda"}, "all"),
        ("hard_initial_dirichlet", {"device": "cuda"}, "all"),
    ],
)
def test_explicit_auto_resolves_supported_scope(
    constraints: str, runtime: dict[str, Any], expected: str
) -> None:
    config = CASES["heat_neumann2d"].config("cpu", 3, 19)
    config["training"].update(backward="auto", constraints=constraints)
    config["runtime"].update(runtime)
    settings = TaskSettings.read(config)
    assert settings.training.backward == "auto"
    assert resolve_backward(settings.training, settings.runtime) == expected
    config["training"]["backward"] = "all"
    settings = TaskSettings.read(config)
    assert resolve_backward(settings.training, settings.runtime) == "all"


def test_explicit_parameters_can_be_experimented_outside_auto_constraint_scope() -> None:
    config = CASES["burgers"].config("cpu", 3, 19)
    config["training"]["backward"] = "parameters"
    settings = TaskSettings.read(config)
    assert resolve_backward(settings.training, settings.runtime) == "parameters"
