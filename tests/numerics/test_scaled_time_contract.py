"""Семантика физических запросов и сохранённых правил приведённого времени."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from benchmarks.quadratic_clock_case import SOURCE, clock_config
from pinnstorch.data import PointCloud
from pinnstorch.models.net.scaled_time import validate_time_scale_checkpoint
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings


@pytest.mark.parametrize("reference", [0, -1, 0.1, float("nan"), float("inf"), True])
def test_invalid_reference_rejected(reference):
    config = clock_config("cpu", 1, 11)
    config["network"]["time_scale"]["reference"] = reference
    with pytest.raises(ValueError):
        TaskSettings.read(config)


@pytest.mark.parametrize(
    "mutation", ["parameter", "negative", "time", "encoding", "hard", "extrapolate", "s"]
)
def test_unsupported_clock_context_rejected(mutation):
    config = clock_config("cpu", 1, 11)
    if mutation == "parameter":
        config["network"]["time_scale"]["parameter"] = "missing"
    elif mutation == "negative":
        config["domain"]["parameters"]["a"]["interval"][0] = -0.1
    elif mutation == "time":
        config["domain"]["time"]["interval"] = [1, 2]
    elif mutation == "encoding":
        config["network"]["spatial_encoding"] = "neumann_cosine"
    elif mutation == "hard":
        config["training"]["constraints"] = "hard_initial_dirichlet"
    elif mutation == "extrapolate":
        config["prediction"] = {"allow_extrapolation": True}
    else:
        config["domain"]["parameters"]["s"] = {"interval": [0, 1], "points": 2}
    with pytest.raises(ValueError):
        TaskSettings.read(config)


def make_model(tmp_path):
    config = clock_config("cpu", 1, 11)
    settings = TaskSettings.read(config)
    factory = SolverFactory(settings, load_problem(SOURCE), tmp_path, SOURCE.parent)
    return config, settings, factory._model(PointCloud(lambda: settings.domain.grid()))


def test_complete_code_coverage_and_saved_settings(tmp_path: Path):
    config, settings, model = make_model(tmp_path)
    rows = config["domain"]["parameter_rows"]
    assert len(rows) == 14 and all(a == 0.15 for a, b in rows)
    assert model.hparams["input_names"] == ("x", "a", "b", "t")
    assert model.hparams["input_bounds"][1] == (0.05, 0.15)
    assert model.net.identity()["network_names"] == ["x", "b", "s"]
    path = tmp_path / "settings.yaml"
    settings.save(path)
    assert TaskSettings.load(path) == settings
    assert yaml.safe_load(path.read_text())["network"]["time_scale"] == {
        "parameter": "a",
        "reference": 0.15,
    }


@pytest.mark.parametrize(
    "damage", ["missing", "reference", "version", "order", "bounds", "normalization"]
)
def test_checkpoint_clock_semantics_reject_before_weights(tmp_path: Path, damage):
    _, _, model = make_model(tmp_path)
    saved = {
        "hyper_parameters": {"input_transform": model.net.identity()},
        "state_dict": deepcopy(model.state_dict()),
    }
    validate_time_scale_checkpoint(saved, model.net)
    if damage == "missing":
        del saved["hyper_parameters"]["input_transform"]
    elif damage == "normalization":
        saved["state_dict"]["net.network.ub"][1] += 1
    else:
        key, value = {
            "reference": ("reference", 0.2),
            "version": ("version", 2),
            "order": ("physical_names", ["x", "b", "a", "t"]),
            "bounds": ("network_bounds", [[0, 1], [1, 3], [0, 1]]),
        }[damage]
        saved["hyper_parameters"]["input_transform"][key] = value
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="Checkpoint"):
        validate_time_scale_checkpoint(saved, model.net)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


@pytest.mark.parametrize("value", [0.049, 0.151, float("nan"), float("inf")])
def test_direct_physical_query_range_guard(tmp_path: Path, value):
    _, _, model = make_model(tmp_path)
    x = torch.ones(2, 1) * 0.5
    with pytest.raises(ValueError, match="physical input 'a'"):
        model([x, torch.full_like(x, value), torch.ones_like(x)], x)
