"""Verify PINN derivatives against analytic results using real CPU tensors."""

import importlib.util
import math
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[2] / "pinnstorch/utils/gradient_fn.py"
spec = importlib.util.spec_from_file_location("gradient_under_test", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
# Load the actual helper without importing the training stack via pinnstorch.__init__.
spec.loader.exec_module(module)
Gradient = Callable[[torch.Tensor, torch.Tensor | list[torch.Tensor]], list[torch.Tensor]]
gradient: Gradient = module.gradient
fwd_gradient: Gradient = module.fwd_gradient


def coordinates() -> torch.Tensor:
    return torch.tensor([[0.2], [0.4], [0.7]], dtype=torch.float64, requires_grad=True)


def test_first_second_and_third_derivatives() -> None:
    x = coordinates()
    first = gradient(x**3, x)[0]
    second = gradient(first, x)[0]
    third = gradient(second, x)[0]

    torch.testing.assert_close(first, 3 * x**2)
    torch.testing.assert_close(second, 6 * x)
    torch.testing.assert_close(third, torch.full_like(x, 6))


def test_derivatives_preserve_gradients_for_training_weights() -> None:
    x = coordinates()
    weight = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    first = gradient(weight * x**3, x)[0]
    second = gradient(first, x)[0]

    loss = second.square().mean()
    loss.backward()

    assert weight.grad is not None
    torch.testing.assert_close(weight.grad, 72 * weight.detach() * x.detach().square().mean())


def test_forward_gradient_keeps_output_components_separate() -> None:
    x, y = coordinates(), coordinates()
    outputs = torch.cat([x**2 + y**3, x**3 + y**2], dim=1)

    dx, dy = fwd_gradient(outputs, [x, y])
    dxx = fwd_gradient(dx, x)[0]

    torch.testing.assert_close(dx, torch.cat([2 * x, 3 * x**2], dim=1))
    torch.testing.assert_close(dy, torch.cat([3 * y**2, 2 * y], dim=1))
    torch.testing.assert_close(dxx, torch.cat([torch.full_like(x, 2), 6 * x], dim=1))


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("coefficient", [0.1, 0.3, 0.5])
def test_heat_equation_residual(dimensions: int, coefficient: float) -> None:
    spatial = [coordinates() for _ in range(dimensions)]
    time = coordinates()
    # Exact solution on the unit box with zero temperature on every face.
    temperature = torch.exp(-coefficient * dimensions * math.pi**2 * time)
    for coordinate in spatial:
        temperature = temperature * torch.sin(math.pi * coordinate)

    derivatives = gradient(temperature, [*spatial, time])
    laplacian = torch.zeros_like(temperature)
    for coordinate, first in zip(spatial, derivatives[:-1]):
        laplacian = laplacian + gradient(first, coordinate)[0]
    residual = derivatives[-1] - coefficient * laplacian

    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=1e-12)
    # Ensure the test distinguishes the correct PDE from an incorrect coefficient.
    wrong_residual = derivatives[-1] - 2 * coefficient * laplacian
    assert wrong_residual.abs().max().item() > 1e-3
