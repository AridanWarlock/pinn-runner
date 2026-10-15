"""Temporal sampling changes the density, not the physical coordinate derivatives."""

from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.cases import CASES
from pinnstorch.api import Coordinates, Equation
from pinnstorch.data import MeshSampler, PINNDataModule, PointCloud
from pinnstorch.data.domains.point_cloud import create_interval_data
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings
from pinnstorch.training.sampling import ResampledLHS, TimePower


def test_time_power_is_inverse_cdf_on_shifted_interval_and_preserves_input() -> None:
    unit = ((np.arange(10000) + 0.5) / 10000).reshape(-1, 1)
    original = -0.3 + 2.1 * unit
    saved = original.copy()
    transform = TimePower((-0.3, 1.8), 1)
    actual = transform(original)
    normalized = (actual + 0.3) / 2.1
    np.testing.assert_allclose(normalized**2, unit, atol=1e-15)
    np.testing.assert_array_equal(original, saved)
    assert float(normalized.mean()) == pytest.approx(2 / 3, abs=1e-6)
    np.testing.assert_array_equal(transform(np.array([[-0.3], [1.8]])), [[-0.3], [1.8]])
    assert TimePower((-0.3, 1.8), 0)(original) is original
    clock = torch.tensor(actual[:11], requires_grad=True)
    equation = Equation({"u": clock**3}, Coordinates({"x": torch.zeros_like(clock)}, clock))
    torch.testing.assert_close(equation.dt("u"), 3 * clock**2)


@pytest.mark.parametrize("power", [-1, float("inf"), float("nan")])
def test_invalid_time_density_is_rejected(power: float) -> None:
    with pytest.raises(ValueError):
        TimePower((0, 1), power)
    config = CASES["heat_neumann3d"].config("cpu", 1, 19)
    config["training"]["collocation"]["time_power"] = power
    with pytest.raises(ValueError):
        TaskSettings.read(config)


def sampler(power: float) -> MeshSampler:
    mesh = PointCloud(lambda: create_interval_data((0, 1), (0.2, 0.8), 3, 3))
    return MeshSampler(
        mesh,
        num_sample=32,
        collection_points=["heat"],
        sampling_policy=ResampledLHS(32, mesh.lb, mesh.ub, 19, 2),
        time_map=TimePower((0.2, 0.8), power),
    )


def test_density_survives_refresh_checkpoint_and_rejects_changed_recipe() -> None:
    first = sampler(1)
    second = sampler(1)
    data = PINNDataModule([first], None, condition_names=["equations"])
    restored = PINNDataModule([second], None, condition_names=["equations"])
    data.refresh_collocation(2)
    state = data.state_dict()
    restored.load_state_dict(state)
    for epoch in (2, 3, 4):
        data.refresh_collocation(epoch)
        restored.refresh_collocation(epoch)
        torch.testing.assert_close(
            first.time_domain_sampled, second.time_domain_sampled, rtol=0, atol=0
        )
        normalized = (first.time_domain_sampled.numpy().reshape(-1) - 0.2) / 0.6
        bins = np.floor(normalized**2 * 32).astype(int)
        np.testing.assert_array_equal(np.sort(bins), np.arange(32))
    mismatch = PINNDataModule([sampler(2)], None, condition_names=["equations"])
    with pytest.raises(ValueError, match="sampling recipe"):
        mismatch.load_state_dict(state)
    assert sampler(0).sampling_signature() == (32, 19, 2)


def test_initial_density_uses_same_spatial_draw_and_no_extra_randomness() -> None:
    np.random.seed(19)
    original = sampler(0)
    after_original = np.random.random(3)
    np.random.seed(19)
    transformed = sampler(1)
    after_transformed = np.random.random(3)
    for left, right in zip(original.spatial_domain_sampled, transformed.spatial_domain_sampled):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    expected = TimePower((0.2, 0.8), 1)(original.time_domain_sampled.numpy())
    np.testing.assert_allclose(transformed.time_domain_sampled.numpy(), expected, atol=1e-7)
    np.testing.assert_array_equal(after_original, after_transformed)


def test_uniform_validation_and_grid_contracts(tmp_path: Path) -> None:
    case = CASES["heat_neumann3d"]
    config = case.config("cpu", 1, 19)
    config["training"]["collocation"] = {"sampling": "grid", "time_power": 1}
    with pytest.raises(ValueError, match="time_power"):
        TaskSettings.read(config)
    config = case.config("cpu", 1, 19)
    config["validation"] = {"sampling": "lhs", "points": 8, "time_power": 1}
    with pytest.raises(ValueError, match="Validation time density"):
        SolverFactory(TaskSettings.read(config), load_problem(case.source), tmp_path).build(False)


@pytest.mark.parametrize("points,power", [(2048, 0), (512, 1)])
def test_sampling_variants_preserve_initial_network_weights(
    tmp_path: Path, points: int, power: float
) -> None:
    case = CASES["heat_neumann3d"]
    config = case.config("cpu", 1, 19)
    config["training"].update(
        constraints="hard_initial_robin", collocation={"sampling": "lhs", "points": 512}
    )
    config["network"]["hidden_layers"] = [8, 8]
    problem = load_problem(case.source)
    original = SolverFactory(
        TaskSettings.read(config), problem, tmp_path, case.source.parent
    ).build(False)
    config["training"]["collocation"].update(points=points, time_power=power)
    changed = SolverFactory(
        TaskSettings.read(config), problem, tmp_path, case.source.parent
    ).build(False)
    for expected, actual in zip(original.model.parameters(), changed.model.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
