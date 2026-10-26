"""Публичное модальное поле: семантика весов и сохранение численного прототипа."""

from copy import deepcopy

import pytest
import torch

from benchmarks.quadratic_modal_case import SOURCE, modal_config
from pinnstorch.data import PointCloud
from pinnstorch.models.net.quadratic_modal import validate_modal_checkpoint
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings


def make_model(tmp_path):
    config = modal_config("cpu", 1, 11)
    settings = TaskSettings.read(config)
    factory = SolverFactory(settings, load_problem(SOURCE), tmp_path, SOURCE.parent)
    return config, settings, factory._model(PointCloud(lambda: settings.domain.grid()))


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "basis",
        "eigenvalues",
        "version",
        "modes",
        "network_names",
        "physical_bounds",
        "coefficient_normalization",
        "initial_rule",
        "source_sha256",
        "lb",
        "ub",
        "mode_numbers",
        "coefficient_scale",
    ],
)
def test_modal_checkpoint_rejects_changed_meaning_before_weights(tmp_path, damage):
    _, _, model = make_model(tmp_path)
    saved = {
        "hyper_parameters": {"field_representation": model.net.identity()},
        "state_dict": deepcopy(model.state_dict()),
    }
    validate_modal_checkpoint(saved, model.net)
    before = deepcopy(model.state_dict())
    if damage == "missing":
        saved["hyper_parameters"].clear()
    elif damage in ("lb", "ub", "mode_numbers", "coefficient_scale"):
        key = (
            f"net.{damage}"
            if damage in ("mode_numbers", "coefficient_scale")
            else f"net.net.{damage}"
        )
        saved["state_dict"][key].reshape(-1)[0] += 1
    else:
        saved["hyper_parameters"]["field_representation"][damage] = "wrong"
    with pytest.raises(ValueError, match="Checkpoint modal"):
        validate_modal_checkpoint(saved, model.net)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "mutation",
    [
        "clock",
        "encoding",
        "hard",
        "extrapolate",
        "space",
        "time",
        "a",
        "b",
        "order",
        "boundary",
        "table",
    ],
)
def test_incompatible_modal_context_rejected(mutation):
    config = modal_config("cpu", 1, 11)
    if mutation == "clock":
        config["network"]["time_scale"] = {"parameter": "a", "reference": 0.15}
    elif mutation == "encoding":
        config["network"]["spatial_encoding"] = "neumann_cosine"
    elif mutation == "hard":
        config["training"]["constraints"] = "hard_initial_dirichlet"
    elif mutation == "extrapolate":
        config["prediction"] = {"allow_extrapolation": True}
    elif mutation == "space":
        config["domain"]["space"]["x"]["interval"] = [0, 2]
    elif mutation == "time":
        config["domain"]["time"]["interval"] = [1, 2]
    elif mutation in ("a", "b"):
        config["domain"]["parameters"][mutation]["interval"][0] = 0
    elif mutation == "order":
        parameters = config["domain"]["parameters"]
        config["domain"]["parameters"] = {key: parameters[key] for key in ("b", "a")}
    elif mutation == "boundary":
        config["boundaries"][0]["value"] = 1
    else:
        config["coefficients"] = {"conductivity": {"bad": True}}
    with pytest.raises(ValueError):
        TaskSettings.read(config)


@pytest.mark.parametrize("value", [0.049, 0.151, float("nan"), float("inf")])
def test_modal_direct_query_outside_declared_physics_rejected(tmp_path, value):
    _, _, model = make_model(tmp_path)
    x = torch.full((2, 1), 0.5)
    with pytest.raises(ValueError, match="physical input 'a'"):
        model([x, torch.full_like(x, value), torch.ones_like(x)], x)


@pytest.mark.parametrize("value", [1, "true", {}, []])
def test_modal_switch_requires_boolean(value):
    config = modal_config("cpu", 1, 11)
    config["network"]["quadratic_modal"] = value
    with pytest.raises(ValueError):
        TaskSettings.read(config)


def test_modal_rejects_additional_problem_transform(tmp_path):
    from dataclasses import replace

    config = modal_config("cpu", 1, 11)
    settings = TaskSettings.read(config)
    problem = replace(load_problem(SOURCE), transform=lambda equation: {"u": equation.field("u")})
    factory = SolverFactory(settings, problem, tmp_path, SOURCE.parent)
    with pytest.raises(ValueError, match="no Problem.transform"):
        factory._model(PointCloud(lambda: settings.domain.grid()))


def test_public_initial_field_is_used_with_physical_derivatives(tmp_path):
    from dataclasses import replace

    _, settings, _ = make_model(tmp_path)
    problem = replace(load_problem(SOURCE), initial=lambda c: {"u": c.x * (1 - c.x)})
    factory = SolverFactory(settings, problem, tmp_path, SOURCE.parent)
    model = factory._model(PointCloud(lambda: settings.domain.grid())).double()
    x = torch.linspace(0, 1, 19, dtype=torch.float64)[:, None].requires_grad_()
    a, b, t = torch.full_like(x, 0.1), torch.full_like(x, 1.5), torch.zeros_like(x)
    actual = model([x, a, b], t)["u"]
    torch.testing.assert_close(actual, x * (1 - x), rtol=0, atol=0)
    derivative = torch.autograd.grad(actual.sum(), x, create_graph=True)[0]
    torch.testing.assert_close(derivative, 1 - 2 * x, rtol=0, atol=2e-15)
    second = torch.autograd.grad(derivative.sum(), x)[0]
    torch.testing.assert_close(second, torch.full_like(x, -2), rtol=0, atol=0)


def test_modal_normalization_requires_modal_field():
    config = modal_config("cpu", 1, 11)
    config["network"].update(quadratic_modal=False, modal_normalization="mode_gap")
    with pytest.raises(ValueError, match="modal_normalization requires"):
        TaskSettings.read(config)


def test_default_public_modal_field_uses_float32_buffers_and_output(tmp_path):
    _, _, model = make_model(tmp_path)
    assert all(value.dtype == torch.float32 for value in model.net.state_dict().values())
    x = torch.full((3, 1), 0.5)
    assert (
        model([x, torch.full_like(x, 0.1), torch.full_like(x, 1.5)], x)["u"].dtype == torch.float32
    )
