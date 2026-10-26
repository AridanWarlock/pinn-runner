"""Публичный модальный путь совпадает с допущенным прототипом и его PDE-градиентами."""

import os
from pathlib import Path

import pytest
import torch
import yaml

from benchmarks.quadratic_modal_case import SOURCE, modal_config
from benchmarks.quadratic_modal_prototype import QuadraticModalPrototype
from pinnstorch.data import PointCloud
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import ProblemAdapter, load_problem
from pinnstorch.runner.settings import TaskSettings


def make_model(tmp_path):
    config = modal_config("cpu", 1, 11)
    settings = TaskSettings.read(config)
    factory = SolverFactory(settings, load_problem(SOURCE), tmp_path, SOURCE.parent)
    return config, settings, factory._model(PointCloud(lambda: settings.domain.grid()))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [11, 29])
def test_public_modal_matches_audited_prototype_and_all_gradients(tmp_path: Path, dtype, seed):
    torch.manual_seed(seed)
    config, settings, model = make_model(tmp_path)
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    model.to(device=device, dtype=dtype)
    prototype = QuadraticModalPrototype().to(device=device, dtype=dtype)
    prototype.load_state_dict(
        {
            key: value
            for key, value in model.net.state_dict().items()
            if key != "coefficient_scale"
        },
        strict=True,
    )
    assert sum(p.numel() for p in model.parameters()) == 3199
    x = torch.linspace(0, 1, 29, dtype=dtype, device=device)[:, None].requires_grad_()
    a = torch.linspace(0.05, 0.15, 29, dtype=dtype, device=device)[:, None]
    b = torch.linspace(1, 2, 29, dtype=dtype, device=device)[:, None]
    t = torch.linspace(0, 1, 29, dtype=dtype, device=device)[:, None].requires_grad_()
    actual = model([x, a, b], t)["u"]
    expected = prototype([x, a, b], t)["u"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    adapter = ProblemAdapter(load_problem(SOURCE), settings.domain)
    residual = adapter.residuals({"u": actual}, x, a, b, t)["heat"]
    reference_residual = adapter.residuals({"u": expected}, x, a, b, t)["heat"]
    torch.testing.assert_close(residual, reference_residual, rtol=0, atol=0)
    for left, right in zip(
        torch.autograd.grad(residual.square().mean(), tuple(model.parameters())),
        torch.autograd.grad(reference_residual.square().mean(), tuple(prototype.parameters())),
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(
        model([x, a, b], torch.zeros_like(t))["u"],
        torch.sin(torch.pi * x) ** 3,
        rtol=0,
        atol=0,
    )
    assert model.hparams["input_bounds"][1] == (0.05, 0.15)
    assert model.hparams["field_representation"]["modes"] == list(range(2, 33))
    path = tmp_path / "config.yaml"
    settings.save(path)
    assert TaskSettings.load(path) == settings
    assert yaml.safe_load(path.read_text())["network"]["quadratic_modal"] is True
    assert config["network"]["hidden_layers"] == [32, 32, 32]
