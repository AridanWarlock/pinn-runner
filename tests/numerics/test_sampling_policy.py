"""Refresh must preserve LHS coverage, random isolation and checkpoint data."""

from pathlib import Path

import numpy as np
import torch

from pinnstorch.data import MeshSampler, PINNDataModule, PointCloud
from pinnstorch.data.domains.point_cloud import create_interval_data
from pinnstorch.training.sampling import ResampledLHS


def make_sampler() -> MeshSampler:
    mesh = PointCloud(lambda: create_interval_data((0, 1), (0, 0.5), 4, 3))
    return MeshSampler(
        mesh,
        num_sample=16,
        collection_points=["pde"],
        sampling_policy=ResampledLHS(16, mesh.lb, mesh.ub, 19, 2),
    )


def test_refresh_is_deterministic_lhs_and_leaves_global_rng_untouched() -> None:
    sampler = make_sampler()
    policy = sampler.sampling_policy
    assert policy is not None
    np.random.seed(77)
    expected = np.random.random(3)
    np.random.seed(77)
    points = policy.points(2)
    np.testing.assert_array_equal(np.random.random(3), expected)
    np.testing.assert_array_equal(points, policy.points(2))
    assert not np.array_equal(points, policy.points(4))
    for dimension, upper in enumerate((1.0, 0.5)):
        bins = np.floor(points[:, dimension] / upper * 16).astype(int)
        np.testing.assert_array_equal(np.sort(bins), np.arange(16))


def test_refresh_cadence_and_datamodule_roundtrip() -> None:
    first = make_sampler()
    data = PINNDataModule([first], None, condition_names=["equations"])
    original = first.time_domain_sampled.clone()
    data.refresh_collocation(1)
    torch.testing.assert_close(first.time_domain_sampled, original)
    data.refresh_collocation(2)
    assert not torch.equal(first.time_domain_sampled, original)
    saved = data.state_dict()
    second = make_sampler()
    restored = PINNDataModule([second], None, condition_names=["equations"])
    restored.load_state_dict(saved)
    torch.testing.assert_close(
        first.time_domain_sampled, second.time_domain_sampled, rtol=0, atol=0
    )
    for epoch in (2, 3, 4):
        data.refresh_collocation(epoch)
        restored.refresh_collocation(epoch)
        torch.testing.assert_close(
            first.time_domain_sampled, second.time_domain_sampled, rtol=0, atol=0
        )
    assert first.last_sampling_epoch == 4


def test_refresh_does_not_change_initial_or_boundary_targets(tmp_path: Path) -> None:
    from benchmarks.cases import CASES
    from pinnstorch.runner.factory import SolverFactory
    from pinnstorch.runner.problem import load_problem
    from pinnstorch.runner.settings import TaskSettings

    case = CASES["advection"]
    config = case.config("cpu", 60, 19)
    config["training"]["collocation"]["resample_every_epochs"] = 1
    objects = SolverFactory(TaskSettings.read(config), load_problem(case.source), tmp_path).build(
        False
    )
    initial, _, boundary = objects.data.train_datasets
    initial_time = initial.time_domain_sampled.clone()
    targets = [value.clone() for value in boundary.solution_sampled]
    objects.data.refresh_collocation(1)
    torch.testing.assert_close(initial.time_domain_sampled, initial_time, rtol=0, atol=0)
    for observed, expected in zip(boundary.solution_sampled, targets):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
