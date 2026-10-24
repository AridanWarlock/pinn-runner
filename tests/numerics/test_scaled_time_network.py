"""Публичная сеть: физические производные и все весовые градиенты."""

import os
from pathlib import Path

import pytest
import torch

from benchmarks.quadratic_clock_case import SOURCE, clock_config
from benchmarks.quadratic_clock_prototype import QuadraticClockPrototype
from pinnstorch.data import PointCloud
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings
from tests.numerics.test_quadratic_clock_prototype import audit_clock_model


class PublicClockModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("reference_scale", torch.tensor(0.15, dtype=torch.float64))

    def forward(self, features, time):
        return self.model(features, time)

    def reference_field(self, x, b, s):
        return self.model([x, torch.ones_like(b) * self.reference_scale, b], s)["u"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [11, 29])
def test_public_scaled_time_derivatives_and_all_weights(dtype, seed, tmp_path: Path):
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    settings = TaskSettings.read(clock_config(device, 1, seed))
    factory = SolverFactory(settings, load_problem(SOURCE), tmp_path, SOURCE.parent)
    torch.manual_seed(seed)
    public = factory._model(PointCloud(lambda: settings.domain.grid()))
    torch.manual_seed(seed)
    prototype = QuadraticClockPrototype().to(device=device, dtype=dtype)
    model = PublicClockModel(public).to(device=device, dtype=dtype).eval()
    for actual, expected in zip(model.parameters(), prototype.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    x = torch.linspace(0.01, 0.99, 13, dtype=dtype, device=device)[:, None]
    with torch.no_grad():
        for a in (0.05, 0.075, 0.125, 0.15):
            features = [x, torch.full_like(x, a), torch.full_like(x, 1.4)]
            torch.testing.assert_close(model(features, x)["u"], prototype(features, x)["u"])
    audit_clock_model(model, dtype, seed, device)
