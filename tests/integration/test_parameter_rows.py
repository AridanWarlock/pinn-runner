"""Whole-function code rows remain coherent in meshes, LHS and queries."""

from copy import deepcopy

import numpy as np
import pytest

from pinnstorch.data import PointCloud, create_interval_data
from pinnstorch.runner.settings import TaskSettings


def row_config():
    return dict(
        version=1,
        domain=dict(
            space={"x": dict(interval=[0, 1], points=3)},
            time=dict(interval=[0, 1], points=2),
            parameters={
                "a": dict(interval=[-0.7, 0.7], points=3),
                "b": dict(interval=[-0.7, 0.7], points=3),
            },
            parameter_rows=[[-0.6, 0.36], [0, 0], [0.6, 0.36]],
        ),
    )


def test_whole_rows_survive_grid_lhs_normalization_and_mutation():
    settings = TaskSettings.read(row_config())
    data = settings.domain.grid()
    mesh = PointCloud(lambda: data)
    assert mesh.spatial_domain_mesh.shape == (9, 2, 3)
    expected = np.tile(np.array([[-0.6, 0.36], [0, 0], [0.6, 0.36]]), (3, 1))
    np.testing.assert_array_equal(mesh.spatial_domain_mesh[:, 0, 1:], expected)
    np.testing.assert_allclose(mesh.lb, [0, -0.7, -0.7, 0])
    np.testing.assert_allclose(mesh.ub, [1, 0.7, 0.7, 1])
    data.parameter_rows[:] = 999
    np.random.seed(29)
    values, clock = mesh.collection_points(257)
    assert values.shape == (257, 3) and clock.shape == (257, 1)
    np.testing.assert_allclose(values[:, 2], values[:, 1] ** 2, atol=0, rtol=0)
    assert len(np.unique(values[:, 1:], axis=0)) == 3
    np.random.seed(29)
    replay, replay_clock = mesh.collection_points(257)
    np.testing.assert_array_equal(replay, values)
    np.testing.assert_array_equal(replay_clock, clock)
    # The old independent LHS is deliberately a negative control for coherence.
    legacy = deepcopy(row_config())
    del legacy["domain"]["parameter_rows"]
    mixed = PointCloud(lambda: TaskSettings.read(legacy).domain.grid()).collection_points(257)[0]
    assert np.max(abs(mixed[:, 2] - mixed[:, 1] ** 2)) > 0.1


def test_many_features_never_reach_cartesian_meshgrid(monkeypatch):
    original = np.meshgrid

    def guarded(*vectors, **kwargs):
        assert len(vectors) <= 3, "code components reached Cartesian meshgrid"
        return original(*vectors, **kwargs)

    monkeypatch.setattr(np, "meshgrid", guarded)
    cfg = row_config()
    cfg["domain"]["parameters"] = {f"z{i}": dict(interval=[-1, 1], points=100) for i in range(24)}
    cfg["domain"]["parameter_rows"] = [[-0.5] * 24, [0] * 24, [0.5] * 24]
    mesh = PointCloud(lambda: TaskSettings.read(cfg).domain.grid())
    assert mesh.spatial_domain_mesh.shape == (9, 2, 25)
    values, _ = mesh.collection_points(127)
    assert all(np.unique(row).size == 1 for row in values[:, 1:])


def test_query_rows_preserve_order_and_original_bounds():
    cfg = row_config()
    cfg["prediction"] = {"parameter_rows": [[0.4, 0.16], [-0.4, 0.16]]}
    settings = TaskSettings.read(cfg)
    data = settings.domain.grid(parameter_rows=settings.prediction.parameter_rows)
    mesh = PointCloud(lambda: data)
    np.testing.assert_array_equal(
        mesh.spatial_domain_mesh[:, 0, 1:], np.tile([[0.4, 0.16], [-0.4, 0.16]], (3, 1))
    )
    np.testing.assert_allclose(mesh.lb, [0, -0.7, -0.7, 0])
    np.testing.assert_allclose(mesh.ub, [1, 0.7, 0.7, 1])


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [[0.1]],
        [[0.1, 0.2], [0.1]],
        [[0.1, 0.2], [0.1, 0.2]],
        [[True, 0.2]],
        [[float("nan"), 0.2]],
        [[float("inf"), 0.2]],
        [[0.8, 0]],
        [[".1", 0.2]],
    ],
)
def test_bad_training_rows_are_rejected(rows):
    cfg = row_config()
    cfg["domain"]["parameter_rows"] = rows
    with pytest.raises(ValueError):
        TaskSettings.read(cfg)


@pytest.mark.parametrize(
    "prediction",
    [
        {"parameters": {"a": [0.1]}},
        {"parameters": {"a": [0.1]}, "parameter_rows": [[0.1, 0.2]]},
        {"parameter_rows": [[0.1]]},
    ],
)
def test_mixed_or_partial_queries_are_rejected(prediction):
    cfg = row_config()
    cfg["prediction"] = prediction
    with pytest.raises(ValueError):
        TaskSettings.read(cfg)


def test_direct_data_api_rejects_unknown_code_columns():
    data = create_interval_data(
        [0, 1],
        [0, 1],
        3,
        2,
        parameters={"a": dict(interval=[-1, 1], points=3)},
        parameter_rows=[[0], [0.5]],
    )
    data.spatial[1][0, 0] = 0.8
    with pytest.raises(ValueError, match="absent"):
        PointCloud(lambda: data)


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_whole_codes_are_separate_from_all_physical_axes(dimensions):
    cfg = row_config()
    for name, interval in list(zip(("y", "z"), ([2, 3], [-4, -2])))[: dimensions - 1]:
        cfg["domain"]["space"][name] = dict(interval=interval, points=2)
    settings = TaskSettings.read(cfg)
    mesh = PointCloud(lambda: settings.domain.grid())
    assert mesh.spatial_dim == dimensions
    assert mesh.spatial_domain_mesh.shape == (9 * 2 ** (dimensions - 1), 2, dimensions + 2)
    points, clock = mesh.collection_points(131)
    assert points.shape == (131, dimensions + 2)
    np.testing.assert_allclose(
        points[:, dimensions + 1], points[:, dimensions] ** 2, atol=0, rtol=0
    )
    for index, axis in enumerate(settings.domain.space.values()):
        assert np.all(points[:, index] >= axis.interval[0])
        assert np.all(points[:, index] <= axis.interval[1])
    assert np.all((clock >= 0) & (clock <= 1))
