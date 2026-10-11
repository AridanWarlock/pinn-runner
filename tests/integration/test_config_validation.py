"""Strict public settings reject errors before constructing the solver."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from pinnstorch.runner.settings import TaskSettings

CONFIG = yaml.safe_load(
    (Path(__file__).parents[2] / "examples/heat_parameterized/config.yaml").read_text()
)


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("network", "hidden_layers"), [True], "positive integers"),
        (("network", "hidden_layers"), [], "nonempty"),
        (("training", "batch_size"), 0, "batch_size"),
        (("training", "epochs"), None, "epochs"),
        (("training", "optimizer", "learning_rate"), -1, "learning_rate"),
        (("runtime", "device"), "gpu", "device"),
        (("runtime", "devices"), [True], "devices"),
        (("runtime", "precision"), "unknown", "precision"),
        (("domain", "space", "x", "interval"), [1, 0], "increasing"),
        (("domain", "space", "x", "points"), True, "points"),
        (("prediction", "parameters"), {"missing": [0.2]}, "Unknown prediction"),
        (("training", "collocation"), {"sampling": "lhs"}, "requires points"),
        (("validation",), {"sampling": "grid", "points": 5}, "does not accept"),
        (("version",), True, "version"),
        (("_target_",), "anything", "unknown options"),
    ],
)
def test_invalid_settings(path: tuple[str, ...], value: object, message: str) -> None:
    source = deepcopy(CONFIG)
    target = source
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        TaskSettings.read(source)


def test_settings_roundtrip(tmp_path: Path) -> None:
    source = deepcopy(CONFIG)
    source["boundaries"].append(
        {
            "kind": "robin",
            "field": "T",
            "axis": "x",
            "alpha": 2,
            "beta": 3,
            "value_function": "environment",
        }
    )
    settings = TaskSettings.read(source)
    path = tmp_path / "config.yaml"
    settings.save(path)
    assert TaskSettings.load(path) == settings
    assert "_target_" not in path.read_text()


@pytest.mark.parametrize(
    "boundary",
    [
        {"kind": "dirichlet", "axis": "x", "field": "T", "alpha": 2},
        {"kind": "robin", "axis": "x", "field": "T", "alpha": 0, "beta": 0},
        {"kind": "neumann", "axis": "z", "field": "T"},
        {"kind": "heat_flux", "axis": "x", "field": "T", "conductivity": "unknown"},
        {"kind": "dirichlet", "axis": "x", "field": "T", "value": 1, "value_function": "wall"},
    ],
)
def test_invalid_boundary_configuration(boundary: dict[str, object]) -> None:
    source = deepcopy(CONFIG)
    source["boundaries"] = [boundary]
    with pytest.raises(ValueError):
        TaskSettings.read(source)
