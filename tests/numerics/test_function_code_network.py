"""Whole correlated codes and every parameter of the physical heat network."""

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
from pinnstorch.training.refinement import FixedObjective


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [11, 29])
def test_function_code_operator_and_every_weight(tmp_path: Path, dtype, seed: int) -> None:
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    assert device in ("cpu", "cuda")
    case = CASES["heat_constant_surrogate"]
    config = case.config(device, 1, seed)
    config["domain"]["parameters"] = {
        name: {"interval": [-0.7, 0.7], "points": 3} for name in ("a", "b")
    }
    config["domain"]["parameter_rows"] = [[-0.6, 0.36], [0, 0], [0.6, 0.36]]
    source = case.source.parents[1] / "heat_function_code/functions.py"
    factory = SolverFactory(
        TaskSettings.read(config), load_problem(source), tmp_path, source.parent
    )
    objects = factory.build(predicting=False)
    model = objects.model.to(device=device, dtype=dtype).eval()
    fixed = FixedObjective.capture(objects.data, model.device)
    assert len(fixed.batches) >= 3
    for spatial, _, _ in fixed.batches.values():
        codes = torch.cat(spatial[1:], dim=1)
        known = torch.tensor(
            [[-0.6, 0.36], [0, 0], [0.6, 0.36]], dtype=codes.dtype, device=codes.device
        )
        assert (codes[:, None, :] == known[None, :, :]).all(dim=2).any(dim=1).all()
    parameters = tuple(model.parameters())
    assert sum(p.numel() for p in parameters) == 2305
    saved = {key: value.detach().clone() for key, value in model.state_dict().items()}
    fingerprint = tensor_fingerprint(saved)
    rng = np.random.default_rng(20261021)
    np.random.seed(20261021)
    features, clock = objects.mesh.collection_points(37)
    points = np.column_stack((features, clock))
    np.testing.assert_allclose(points[:, 2], points[:, 1] ** 2, atol=0, rtol=0)

    def evaluate(query, wrong=False):
        x, a, b, t = [
            torch.tensor(query[:, i : i + 1], dtype=dtype, device=device, requires_grad=True)
            for i in range(4)
        ]
        output = model([x, a, b], t)
        e = Equation(output, factory.adapter.context([x, a, b], t))
        physical = factory.adapter.residuals(output, x, a, b, t)["heat"]
        k = torch.exp(a * torch.cos(torch.pi * x) + b * torch.cos(2 * torch.pi * x)) / 8
        kx = k * (
            -torch.pi * a * torch.sin(torch.pi * x)
            - 2 * torch.pi * b * torch.sin(2 * torch.pi * x)
        )
        expanded = e.dt("u") - k * e.diff("u", "x", 2) - kx * e.diff("u", "x")
        if wrong:
            expanded = expanded - k * (e.diff("u", "a", 2) + e.diff("u", "b", 2))
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
    xx, aa, bb, tt = [
        torch.tensor(points[:, i : i + 1], dtype=dtype, device=device, requires_grad=True)
        for i in range(4)
    ]
    out = model([xx, aa, bb], tt)
    equation = Equation(out, factory.adapter.context([xx, aa, bb], tt))
    kk = torch.exp(aa * torch.cos(torch.pi * xx) + bb * torch.cos(2 * torch.pi * xx)) / 8
    missing = equation.dt("u") - kk * equation.diff("u", "x", 2)
    assert (missing - residual).detach().abs().max() > 1e-6
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

    a = torch.tensor([[-0.6], [0], [0.6], [0.4]], dtype=dtype, device=device)
    x = torch.tensor([[0.13], [0.37], [0.61], [0.87]], dtype=dtype, device=device)
    b = a.square()
    zero, one = torch.zeros_like(a), torch.ones_like(a)
    with torch.no_grad():
        torch.testing.assert_close(
            model([x, a, b], zero)["u"], torch.sin(torch.pi * x), atol=atol, rtol=rtol
        )
        for endpoint in (zero, one):
            assert model([endpoint, a, b], one)["u"].abs().max() <= atol
    assert tensor_fingerprint(model.state_dict()) == fingerprint
