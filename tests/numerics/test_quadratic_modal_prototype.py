"""Новая нейросеть: физический остаток, все 3199 градиентов и точные IC/BC."""

import os

import numpy as np
import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.fingerprints import tensor_fingerprint
from benchmarks.quadratic_modal_prototype import QuadraticModalPrototype
from pinnstorch.runner.problem import ProblemAdapter, load_problem
from pinnstorch.runner.settings import TaskSettings


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [11, 29])
def test_modal_physical_residual_all_weights_and_boundaries(dtype, seed):
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    assert device in ("cpu", "cuda")
    torch.manual_seed(seed)
    model = QuadraticModalPrototype().to(device=device, dtype=dtype).eval()
    assert sum(p.numel() for p in model.parameters()) == 3199
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    fingerprint = tensor_fingerprint(before)
    case = CASES["heat_quadratic_shapes"]
    adapter = ProblemAdapter(
        load_problem(case.source), TaskSettings.read(case.config("cpu", 1, seed)).domain
    )
    rng = np.random.default_rng(seed)
    query = np.column_stack(
        [
            rng.uniform(0.01, 0.99, 53),
            rng.uniform(0.05, 0.15, 53),
            rng.uniform(1, 2, 53),
            rng.uniform(0.02, 0.98, 53),
        ]
    )
    parameters = tuple(model.parameters())

    def evaluate(points):
        x, a, b, t = [
            torch.tensor(points[:, i : i + 1], dtype=dtype, device=device, requires_grad=True)
            for i in range(4)
        ]
        output = model([x, a, b], t)
        residual = adapter.residuals(output, x, a, b, t)["heat"]
        phi, eigen, first = model.mode_data(x, a, b)
        sn, cs = torch.sin(torch.pi * x), torch.cos(torch.pi * x)
        f = sn**3
        fx = 3 * torch.pi * sn**2 * cs
        fxx = 6 * torch.pi**2 * sn * cs**2 - 3 * torch.pi**2 * sn**3
        operator = a * (1 + x / b) ** 2 * fxx + 2 * a / b * (1 + x / b) * fx
        expected = torch.exp(-first * t) * (
            -(operator + first * f)
            + (model.coefficients(b) * (first - eigen) * phi).sum(dim=1, keepdim=True)
        )
        atol, rtol = (2e-3, 3e-4) if dtype == torch.float32 else (2e-10, 2e-11)
        torch.testing.assert_close(residual, expected, atol=atol, rtol=rtol)
        wrong = expected + first * f * torch.exp(-first * t)
        assert (wrong - residual).abs().max() > 1e-2
        return output["u"], residual, expected

    value, residual, control = evaluate(query)
    actual = torch.cat(
        [
            g.flatten()
            for g in torch.autograd.grad(residual.square().mean(), parameters, retain_graph=True)
        ]
    )
    expected = torch.cat(
        [g.flatten() for g in torch.autograd.grad(control.square().mean(), parameters)]
    )
    atol, rtol = (0.1, 4e-4) if dtype == torch.float32 else (2e-8, 2e-10)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    assert torch.isfinite(actual).all() and actual.norm() > 1
    order = rng.permutation(len(query))
    torch.testing.assert_close(
        evaluate(query[order])[0][np.argsort(order)], value, atol=atol, rtol=rtol
    )
    if dtype == torch.float64:
        direction = torch.tensor(rng.normal(size=actual.numel()), dtype=dtype, device=device)
        direction /= direction.norm()
        exact = float(actual @ direction)
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
                losses.append(float(evaluate(query)[1].square().mean().detach()))
            np.testing.assert_allclose(
                (losses[0] - losses[1]) / (2 * h), exact, atol=1e-5, rtol=1e-5
            )
        model.load_state_dict(before)
    x, a, b, t = [torch.tensor(query[:, i : i + 1], dtype=dtype, device=device) for i in range(4)]
    torch.testing.assert_close(
        model([x, a, b], torch.zeros_like(t))["u"], torch.sin(torch.pi * x) ** 3, atol=0, rtol=0
    )
    for endpoint in (0.0, 1.0):
        boundary = torch.full_like(x, endpoint, requires_grad=True)
        output = model([boundary, a, b], t)["u"]
        assert output.abs().max() < 1e-6
        derivative = torch.autograd.grad(output.sum(), boundary)[0]
        length = torch.log1p(1 / b)
        r = 1 + endpoint / b
        n = model.mode_numbers
        phi_dx = torch.pi * n / (b * length * r**1.5)
        if endpoint == 1:
            phi_dx = phi_dx * torch.where(n.to(torch.int64) % 2 == 0, 1, -1)
        _, eigen, first = model.mode_data(boundary, a, b)
        expected_dx = (
            model.coefficients(b) * phi_dx * (torch.exp(-eigen * t) - torch.exp(-first * t))
        ).sum(dim=1, keepdim=True)
        torch.testing.assert_close(
            derivative,
            expected_dx,
            atol=5e-5 if dtype == torch.float32 else 1e-12,
            rtol=3e-5 if dtype == torch.float32 else 1e-12,
        )
    assert tensor_fingerprint(model.state_dict()) == fingerprint
