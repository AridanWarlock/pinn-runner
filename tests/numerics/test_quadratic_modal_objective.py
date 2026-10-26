"""Интегральная PDE-потеря: независимая физическая квадратура и все градиенты."""

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.quadratic_modal_case import SOURCE, modal_config
from benchmarks.quadratic_modal_objective import ModalResidualObjective
from pinnstorch.data import PointCloud
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import ProblemAdapter, load_problem
from pinnstorch.runner.settings import TaskSettings


@pytest.mark.parametrize("normalization", ["identity", "mode_gap"])
@pytest.mark.parametrize("seed", [11, 29])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_integral_modal_loss_matches_physical_pde_all_gradients(
    tmp_path: Path, seed, dtype, normalization
):
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    torch.manual_seed(seed)
    config = modal_config("cpu", 1, seed)
    config["network"]["modal_normalization"] = normalization
    settings = TaskSettings.read(config)
    problem = load_problem(SOURCE)
    adapter = ProblemAdapter(problem, settings.domain)
    factory = SolverFactory(settings, problem, tmp_path, SOURCE.parent)
    model = factory._model(PointCloud(lambda: settings.domain.grid())).to(
        device=device, dtype=dtype
    )
    initial = adapter.initial_tensors
    forms = [1.0, 1.37, 2.0]
    objective = ModalResidualObjective(initial, forms).to(device=device, dtype=dtype)
    refined = ModalResidualObjective(initial, forms, quadrature=512).to(device=device, dtype=dtype)
    torch.testing.assert_close(objective.targets, refined.targets, atol=2e-13, rtol=0)
    torch.testing.assert_close(
        objective.truncation_floor, refined.truncation_floor, atol=1e-14, rtol=1e-7
    )
    before = deepcopy(model.state_dict())
    coefficients = model.net.coefficients(objective.forms)
    modal_loss = objective.full_integral(coefficients)
    parameters = tuple(model.parameters())
    modal_gradients = torch.cat([g.flatten() for g in torch.autograd.grad(modal_loss, parameters)])
    # Другой порядок квадратуры; остаток вычисляется из производных полного поля.
    xn, xw = np.polynomial.legendre.leggauss(128)
    tn, tw = np.polynomial.legendre.leggauss(16)
    xgrid, tgrid = np.meshgrid((xn + 1) / 2, (tn + 1) / 2, indexing="ij")
    weights = torch.tensor(np.outer(xw / 2, tw / 2).reshape(-1, 1), dtype=dtype, device=device)
    integrated = []
    for shape in forms:
        x = torch.tensor(xgrid.reshape(-1, 1), dtype=dtype, device=device, requires_grad=True)
        t = torch.tensor(tgrid.reshape(-1, 1), dtype=dtype, device=device, requires_grad=True)
        a, b = torch.full_like(x, 0.15), torch.full_like(x, shape)
        outputs = model([x, a, b], t)
        residual = adapter.residuals(outputs, x, a, b, t)["heat"]
        integrated.append((weights * residual.square()).sum())
    physical_loss = torch.stack(integrated).mean()
    physical_gradients = torch.cat(
        [g.flatten() for g in torch.autograd.grad(physical_loss, parameters)]
    )
    atol, rtol = (3e-3, 3e-5) if dtype == torch.float32 else (3e-8, 3e-10)
    torch.testing.assert_close(modal_loss, physical_loss, atol=atol, rtol=rtol)
    gradient_error = modal_gradients - physical_gradients
    relative_l2 = gradient_error.norm() / modal_gradients.norm()
    relative_max = gradient_error.abs().max() / modal_gradients.abs().max()
    gradient_tolerance = 1e-5 if dtype == torch.float32 else 3e-10
    assert relative_l2 < gradient_tolerance
    assert relative_max < gradient_tolerance
    assert modal_gradients.numel() == 3199 and modal_gradients.norm() > 1
    # Переставленные коэффициенты дают другую потерю: порядок мод существенен.
    wrong = objective.full_integral(coefficients.flip(1))
    assert abs(float((wrong - modal_loss).detach())) > 1
    if dtype == torch.float64:
        rng = np.random.default_rng(seed)
        direction = torch.tensor(rng.normal(size=3199), dtype=dtype, device=device)
        direction /= direction.norm()
        exact = float(modal_gradients @ direction)
        for h in (1e-5, 5e-6):
            losses = []
            for sign in (1, -1):
                model.load_state_dict(before)
                with torch.no_grad():
                    offset = 0
                    for parameter in parameters:
                        n = parameter.numel()
                        parameter.add_(
                            direction[offset : offset + n].reshape_as(parameter), alpha=sign * h
                        )
                        offset += n
                losses.append(float(objective(model.net.coefficients(objective.forms)).detach()))
            np.testing.assert_allclose(
                (losses[0] - losses[1]) / (2 * h), exact, rtol=1e-5, atol=1e-5
            )
        model.load_state_dict(before)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
