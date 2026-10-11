"""Check the actual FCN and a short PINN optimization on CPU."""

import importlib.util
import math
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_module(relative_path: str, name: str) -> ModuleType:
    """Load a production module without importing the training stack."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


net_module = load_module("pinnstorch/models/net/neural_net.py", "network_under_test")
grad_module = load_module("pinnstorch/utils/gradient_fn.py", "network_gradients_under_test")
gradient: Callable[
    [torch.Tensor, torch.Tensor | list[torch.Tensor]], list[torch.Tensor]
] = grad_module.gradient


def make_network(dimensions: int, outputs: list[str]) -> torch.nn.Module:
    return net_module.FCN(
        layers=[dimensions + 1, 16, 16, len(outputs)],
        lb=[0.0] * (dimensions + 1),
        ub=[1.0] * (dimensions + 1),
        output_names=outputs,
    )


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_network_outputs_and_coordinate_derivatives(dimensions: int) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        net = make_network(dimensions, ["temperature", "pressure"])
    spatial = [
        torch.linspace(0.1 + 0.1 * axis, 0.7 + 0.1 * axis, 5)
        .reshape(-1, 1)
        .requires_grad_()
        for axis in range(dimensions)
    ]
    time = torch.linspace(0.0, 0.5, 5).reshape(-1, 1).requires_grad_()

    outputs: dict[str, torch.Tensor] = net(spatial, time)

    assert set(outputs) == {"temperature", "pressure"}
    assert not torch.allclose(outputs["temperature"], outputs["pressure"])
    for output in outputs.values():
        assert output.shape == (5, 1)
        assert output.device.type == "cpu"
        assert torch.isfinite(output).all()
        first = gradient(output, [*spatial, time])
        for coordinate, derivative in zip([*spatial, time], first):
            second = gradient(derivative, coordinate)[0]
            assert second.shape == output.shape
            assert torch.isfinite(derivative).all()
            assert torch.isfinite(second).all()


def heat_loss(net: torch.nn.Module, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Combine PDE residual and initial condition for T_t = 0.3 T_xx."""
    x = x.detach().requires_grad_()
    time = time.detach().requires_grad_()
    outputs: dict[str, torch.Tensor] = net([x], time)
    dx, dt = gradient(outputs["temperature"], [x, time])
    dxx = gradient(dx, x)[0]
    residual = dt - 0.3 * dxx
    initial: dict[str, torch.Tensor] = net([x], torch.zeros_like(time))
    initial_error = initial["temperature"] - torch.sin(math.pi * x)
    return residual.square().mean() + initial_error.square().mean()


def test_short_cpu_training_updates_weights_and_reduces_loss() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        net = make_network(1, ["temperature"])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    x = torch.linspace(0.1, 0.9, 16).reshape(-1, 1)
    time = torch.linspace(0.0, 0.5, 16).reshape(-1, 1)
    initial_loss = heat_loss(net, x, time).detach()
    original_weights = [parameter.detach().clone() for parameter in net.parameters()]

    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = heat_loss(net, x, time)
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in net.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        optimizer.step()

    final_loss = heat_loss(net, x, time).detach()
    assert final_loss < initial_loss
    assert any(
        not torch.equal(before, after)
        for before, after in zip(original_weights, net.parameters())
    )
