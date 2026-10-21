"""Physical operator and every weight of the shared constant-k network."""

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
def test_shared_constant_operator_and_every_weight(tmp_path: Path, dtype, seed: int) -> None:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    assert device in ("cpu", "cuda")
    case = CASES["heat_constant_surrogate"]
    factory = SolverFactory(
        TaskSettings.read(case.config(device, 1, seed)),
        load_problem(case.source),
        tmp_path,
        case.source.parent,
    )
    model = factory.build(predicting=False).model.to(device=device, dtype=dtype).eval()
    parameters = tuple(model.parameters())
    assert sum(p.numel() for p in parameters) == 2273
    saved = {key: value.detach().clone() for key, value in model.state_dict().items()}
    fingerprint = tensor_fingerprint(saved)
    rng = np.random.default_rng(20261021)
    points = rng.uniform(size=(37, 3))
    points[:, 1] = 0.05 + 0.2 * points[:, 1]

    def evaluate(query, wrong=False):
        x, k, t = [
            torch.tensor(query[:, i : i + 1], dtype=dtype, device=device, requires_grad=True)
            for i in range(3)
        ]
        output = model([x, k], t)
        e = Equation(output, factory.adapter.context([x, k], t))
        physical = factory.adapter.residuals(output, x, k, t)["heat"]
        expanded = e.dt("u") - k * e.diff("u", "x", 2)
        if wrong:
            expanded = expanded - k * e.diff("u", "k", 2)
        return output["u"], physical, expanded

    value, residual, expanded = evaluate(points)
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
    wrong = evaluate(points, wrong=True)[2]
    assert (wrong - residual).detach().abs().max() > 1e-6
    order = rng.permutation(len(points))
    permuted = evaluate(points[order])
    torch.testing.assert_close(permuted[0][np.argsort(order)], value, atol=atol, rtol=rtol)
    torch.testing.assert_close(permuted[1][np.argsort(order)], residual, atol=atol, rtol=rtol)
    chunk_gradient = torch.zeros_like(gradient)
    for chunk in np.array_split(points, 3):
        chunk_residual = evaluate(chunk)[1]
        chunk_gradient += (
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(chunk_residual.square().mean(), parameters)
                ]
            )
            * len(chunk)
            / len(points)
        )
    torch.testing.assert_close(chunk_gradient, gradient, atol=atol, rtol=rtol)

    if dtype == torch.float64:
        directions = [torch.ones_like(p) for p in parameters]
        norm = sum(d.numel() for d in directions) ** 0.5
        directions = [d / norm for d in directions]
        directional = float(
            sum(
                (g * d.flatten()).sum()
                for g, d in zip(torch.split(gradient, [p.numel() for p in parameters]), directions)
            )
        )
        for h in (1e-5, 5e-6):
            losses = []
            for sign in (1, -1):
                model.load_state_dict(saved)
                with torch.no_grad():
                    for p, d in zip(parameters, directions):
                        p.add_(d, alpha=sign * h)
                losses.append(float(evaluate(points)[1].square().mean().detach()))
            np.testing.assert_allclose(
                (losses[0] - losses[1]) / (2 * h), directional, atol=1e-9, rtol=1e-4
            )
        model.load_state_dict(saved)

    k = torch.tensor([[0.05], [0.1], [0.2], [0.25]], dtype=dtype, device=device)
    x = torch.tensor([[0.13], [0.37], [0.61], [0.87]], dtype=dtype, device=device)
    zero, one = torch.zeros_like(k), torch.ones_like(k)
    with torch.no_grad():
        torch.testing.assert_close(
            model([x, k], zero)["u"], torch.sin(torch.pi * x), atol=atol, rtol=rtol
        )
        for endpoint in (zero, one):
            assert model([endpoint, k], one)["u"].abs().max() <= atol
    assert tensor_fingerprint(model.state_dict()) == fingerprint
