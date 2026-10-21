"""Физический оператор и все 2305 весовых производных новой постановки."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.fingerprints import tensor_fingerprint
from pinnstorch.api import Equation
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [11, 29])
def test_quadratic_shape_operator_and_every_weight(tmp_path: Path, dtype, seed: int) -> None:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    assert device in ("cpu", "cuda")
    case = CASES["heat_constant_surrogate"]
    config = case.config(device, 1, seed)
    config["domain"]["parameters"] = {
        "a": {"interval": [0.05, 0.15], "points": 3},
        "b": {"interval": [1, 2], "points": 3},
    }
    pool = [[0.05, 1], [0.05, 2], [0.15, 1], [0.15, 2], [0.1, 1.5]]
    config["domain"]["parameter_rows"] = pool
    source = case.source.parents[1] / "heat_quadratic_shapes/functions.py"
    factory = SolverFactory(
        TaskSettings.read(config), load_problem(source), tmp_path, source.parent
    )
    objects = factory.build(predicting=False)
    model = objects.model.to(device=device, dtype=dtype).eval()
    parameters = tuple(model.parameters())
    assert sum(p.numel() for p in parameters) == 2305
    saved = {key: value.detach().clone() for key, value in model.state_dict().items()}
    fingerprint = tensor_fingerprint(saved)
    rng = np.random.default_rng(seed)
    points = np.column_stack(
        (
            rng.uniform(0.01, 0.99, 37),
            np.asarray(pool)[rng.integers(0, len(pool), 37)],
            rng.uniform(0.01, 0.99, 37),
        )
    )

    def evaluate(query):
        x, a, b, t = [
            torch.tensor(query[:, i : i + 1], dtype=dtype, device=device, requires_grad=True)
            for i in range(4)
        ]
        out = model([x, a, b], t)
        equation = Equation(out, factory.adapter.context([x, a, b], t))
        residual = factory.adapter.residuals(out, x, a, b, t)["heat"]
        k = a * (1 + x / b) ** 2
        kx = 2 * a / b * (1 + x / b)
        expanded = equation.dt("u") - k * equation.diff("u", "x", 2) - kx * equation.diff("u", "x")
        wrong = expanded - k * (equation.diff("u", "a", 2) + equation.diff("u", "b", 2))
        missing = equation.dt("u") - k * equation.diff("u", "x", 2)
        return out["u"], residual, expanded, wrong, missing

    value, residual, expanded, wrong, missing = evaluate(points)
    gradient = torch.cat(
        [
            g.flatten()
            for g in torch.autograd.grad(residual.square().mean(), parameters, retain_graph=True)
        ]
    )
    control = torch.cat(
        [g.flatten() for g in torch.autograd.grad(expanded.square().mean(), parameters)]
    )
    atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (1e-10, 1e-9)
    torch.testing.assert_close(residual, expanded, atol=atol, rtol=rtol)
    torch.testing.assert_close(gradient, control, atol=atol, rtol=rtol)
    assert torch.isfinite(gradient).all()
    assert (wrong - residual).abs().max() > 1e-6
    assert (missing - residual).abs().max() > 1e-6
    order = rng.permutation(len(points))
    permuted = evaluate(points[order])
    torch.testing.assert_close(permuted[0][np.argsort(order)], value, atol=atol, rtol=rtol)
    torch.testing.assert_close(permuted[1][np.argsort(order)], residual, atol=atol, rtol=rtol)
    if dtype == torch.float64:
        direction = torch.ones_like(gradient) / gradient.numel() ** 0.5
        exact = float(gradient @ direction)
        for step in [1e-5, 5e-6]:
            losses = []
            for sign in [1, -1]:
                model.load_state_dict(saved)
                with torch.no_grad():
                    offset = 0
                    for parameter in parameters:
                        count = parameter.numel()
                        parameter.add_(
                            direction[offset : offset + count].reshape_as(parameter),
                            alpha=sign * step,
                        )
                        offset += count
                losses.append(float(evaluate(points)[1].square().mean().detach()))
            np.testing.assert_allclose(
                (losses[0] - losses[1]) / (2 * step), exact, rtol=1e-4, atol=1e-9
            )
        model.load_state_dict(saved)
    x, a, b, t = [torch.tensor(points[:, i : i + 1], dtype=dtype, device=device) for i in range(4)]
    with torch.no_grad():
        torch.testing.assert_close(
            model([x, a, b], torch.zeros_like(t))["u"],
            torch.sin(torch.pi * x) ** 3,
            atol=atol,
            rtol=rtol,
        )
        for endpoint in [torch.zeros_like(x), torch.ones_like(x)]:
            assert model([endpoint, a, b], t)["u"].abs().max() <= atol
    assert tensor_fingerprint(model.state_dict()) == fingerprint
