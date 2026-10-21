"""Физические часы: производные, все 2273 градиента и конечные разности."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.fingerprints import tensor_fingerprint
from benchmarks.quadratic_clock_prototype import QuadraticClockPrototype
from pinnstorch.api import Coordinates, Equation
from pinnstorch.runner.problem import ProblemAdapter, load_problem
from pinnstorch.runner.settings import TaskSettings


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [11, 29])
def test_clock_chain_rule_and_all_weights(dtype, seed, tmp_path: Path):
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    assert device in ("cpu", "cuda")
    torch.manual_seed(seed)
    model = QuadraticClockPrototype().to(device=device, dtype=dtype).eval()
    assert sum(p.numel() for p in model.parameters()) == 2273
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    fingerprint = tensor_fingerprint(before)
    case = CASES["heat_quadratic_shapes"]
    adapter = ProblemAdapter(
        load_problem(case.source), TaskSettings.read(case.config("cpu", 1, seed)).domain
    )
    parameters = tuple(model.parameters())
    rng = np.random.default_rng(seed)
    query = np.column_stack(
        (
            rng.uniform(0.01, 0.99, 37),
            rng.uniform(0.05, 0.15, 37),
            rng.uniform(1, 2, 37),
            rng.uniform(0.01, 0.99, 37),
        )
    )

    def evaluate(points):
        x, a, b, t = [
            torch.tensor(points[:, i : i + 1], dtype=dtype, device=device, requires_grad=True)
            for i in range(4)
        ]
        physical = model([x, a, b], t)
        residual = adapter.residuals(physical, x, a, b, t)["heat"]
        original = Equation(physical, adapter.context([x, a, b], t))
        scale = a / model.reference_scale
        s = (scale * t).detach().clone().requires_grad_()
        value = model.reference_field(x, b, s)
        reference = Equation({"u": value}, Coordinates({"x": x}, s, {"b": b}))
        q = (1 + x / b) ** 2
        qx = 2 / b * (1 + x / b)
        ref_residual = reference.dt("u") - model.reference_scale * (
            q * reference.diff("u", "x", 2) + qx * reference.diff("u", "x")
        )
        atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (2e-12, 2e-11)
        torch.testing.assert_close(
            original.dt("u"), scale * reference.dt("u"), atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            original.diff("u", "x", 2), reference.diff("u", "x", 2), atol=atol, rtol=rtol
        )
        torch.testing.assert_close(residual, scale * ref_residual, atol=atol, rtol=rtol)
        wrong = reference.dt("u") - a * (
            q * reference.diff("u", "x", 2) + qx * reference.diff("u", "x")
        )
        assert (wrong - residual).abs().max() > 1e-3
        return physical["u"], residual, scale * ref_residual

    value, residual, control = evaluate(query)
    gradient = torch.cat(
        [
            g.flatten()
            for g in torch.autograd.grad(residual.square().mean(), parameters, retain_graph=True)
        ]
    )
    expected = torch.cat(
        [g.flatten() for g in torch.autograd.grad(control.square().mean(), parameters)]
    )
    atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (2e-11, 2e-10)
    torch.testing.assert_close(gradient, expected, atol=atol, rtol=rtol)
    assert torch.isfinite(gradient).all()
    order = rng.permutation(len(query))
    torch.testing.assert_close(
        evaluate(query[order])[0][np.argsort(order)], value, atol=atol, rtol=rtol
    )
    if dtype == torch.float64:
        direction = torch.ones_like(gradient) / gradient.numel() ** 0.5
        exact = float(gradient @ direction)
        for h in [1e-5, 5e-6]:
            losses = []
            for sign in [1, -1]:
                model.load_state_dict(before)
                with torch.no_grad():
                    offset = 0
                    for parameter in parameters:
                        n = parameter.numel()
                        parameter.add_(
                            direction[offset : offset + n].reshape_as(parameter), alpha=sign * h
                        )
                        offset += n
                losses.append(float(evaluate(query)[1].square().mean().detach()))
            np.testing.assert_allclose(
                (losses[0] - losses[1]) / (2 * h), exact, rtol=1e-4, atol=1e-9
            )
        model.load_state_dict(before)
    x, a, b, t = [torch.tensor(query[:, i : i + 1], dtype=dtype, device=device) for i in range(4)]
    with torch.no_grad():
        torch.testing.assert_close(
            model([x, a, b], torch.zeros_like(t))["u"],
            torch.sin(torch.pi * x) ** 3,
            atol=atol,
            rtol=rtol,
        )
        for endpoint in [torch.zeros_like(x), torch.ones_like(x)]:
            assert model([endpoint, a, b], t)["u"].abs().max() < atol
    assert tensor_fingerprint(model.state_dict()) == fingerprint
