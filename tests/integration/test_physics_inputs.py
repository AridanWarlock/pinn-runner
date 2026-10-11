"""Test coordinate-only meshes and explicitly supplied initial conditions."""

import numpy as np
import pytest
from numpy.typing import NDArray

from pinnstorch.data import InitialCondition, PointCloud, create_interval_data


def make_mesh() -> PointCloud:
    return PointCloud(read_data_fn=lambda: create_interval_data([0.0, 1.0], [2.0, 3.0], 4, 3))


def test_grid_has_coordinates_and_no_reference_values() -> None:
    mesh = make_mesh()
    spatial, time, values = mesh.flatten_mesh(None)

    assert mesh.solution == {}
    assert values == {}
    assert spatial.shape == (12, 1)
    assert time.shape == (12, 1)
    np.testing.assert_array_equal(spatial[:, 0], np.repeat(np.linspace(0, 1, 4), 3))
    np.testing.assert_array_equal(time[:, 0], np.tile(np.linspace(2, 3, 3), 4))
    np.testing.assert_array_equal(mesh.lb, [0, 2])
    np.testing.assert_array_equal(mesh.ub, [1, 3])


def test_initial_values_are_taken_from_user_function() -> None:
    def initial(spatial: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        return {"T": 10 + spatial[:, 0:1] ** 2}

    condition = InitialCondition(make_mesh(), solution=["T"], initial_fun=initial)
    spatial, time, target = condition[:]

    np.testing.assert_allclose(target["T"].numpy(), 10 + spatial[0].numpy() ** 2)
    np.testing.assert_array_equal(time.numpy(), np.full((4, 1), 2.0))


@pytest.mark.parametrize("failure", ["missing", "wrong_shape", "non_finite"])
def test_rejects_invalid_initial_values(failure: str) -> None:
    def initial(spatial: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        if failure == "missing":
            return {}
        if failure == "wrong_shape":
            return {"T": np.zeros(len(spatial))}
        return {"T": np.full((len(spatial), 1), np.nan)}

    with pytest.raises(ValueError, match="initial_conditions"):
        InitialCondition(make_mesh(), solution=["T"], initial_fun=initial)


@pytest.mark.parametrize(
    ("bounds", "count"),
    [([1.0, 0.0], 4), ([0.0, 0.0], 4), ([0.0, np.inf], 4), ([0.0, 1.0], 1)],
)
def test_rejects_invalid_domain(bounds: list[float], count: int) -> None:
    with pytest.raises(ValueError):
        create_interval_data(bounds, [0.0, 1.0], count, 3)
