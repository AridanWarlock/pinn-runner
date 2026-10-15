"""Independent quadrature distinguishes all slow modes from a fast remainder."""

import numpy as np
import pytest

from benchmarks.cases import Array
from benchmarks.neumann_projection import project_neumann_error


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_gauss_projection_recovers_prescribed_error_energies(dimensions: int) -> None:
    rules = [np.polynomial.legendre.leggauss(17) for _ in range(dimensions)]
    bounds = tuple((-0.3 + i * 0.1, 1.2 + i * 0.2) for i in range(dimensions))
    nodes = tuple(
        lower + (rule[0] + 1) * (upper - lower) / 2 for rule, (lower, upper) in zip(rules, bounds)
    )
    weights = tuple(rule[1] * (upper - lower) / 2 for rule, (lower, upper) in zip(rules, bounds))
    mesh = np.meshgrid(*nodes, indexing="ij")
    count = int(np.prod([len(n) for n in nodes]))
    error = np.full((count, 2), 0.2)

    def phi(index: tuple[int, ...]) -> Array:
        result = np.ones(count)
        for k, column, (lower, upper) in zip(index, mesh, bounds):
            result *= np.cos(k * np.pi * (column.reshape(-1) - lower) / (upper - lower))
        return result

    original = tuple(1 for _ in nodes)
    fast = (3, *original[1:])
    error += phi(original)[:, None] * np.array([[0.3, -0.4]])
    error += phi(fast)[:, None] * np.array([[0.1, 0.2]])
    reference = 0.1 * np.pi**2 * sum(1 / (upper - lower) ** 2 for lower, upper in bounds)
    result = project_neumann_error(nodes, weights, bounds, error, reference)
    volume = float(np.prod([b - a for a, b in bounds]))
    np.testing.assert_allclose(result.groups["constant"], 0.04 * volume, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(
        result.groups["same_rate"],
        np.array([0.09, 0.16]) * volume / (2**dimensions),
        rtol=1e-10,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        result.groups["faster"],
        np.array([0.01, 0.04]) * volume / (2**dimensions),
        rtol=1e-10,
        atol=1e-13,
    )
    np.testing.assert_allclose(result.groups["slower_nonconstant"], 0, atol=1e-13)
    assert result.gram_defect < 1e-10
    assert result.normalization_defect < 1e-10
    assert result.reconstruction_max_error < 1e-11


def test_long_axis_has_slow_modes_with_index_greater_than_one() -> None:
    z, w = np.polynomial.legendre.leggauss(17)
    nodes = ((z + 1) * 1.5, (z + 1) / 2)
    weights = (w * 1.5, w / 2)
    bounds = ((0.0, 3.0), (0.0, 1.0))
    mesh = np.meshgrid(*nodes, indexing="ij")
    error = np.cos(3 * np.pi * mesh[0].reshape(-1, 1) / 3)
    reference = 0.1 * np.pi**2 * (1 / 9 + 1)
    result = project_neumann_error(nodes, weights, bounds, error, reference)
    assert (3, 0) in result.indices
    np.testing.assert_allclose(result.groups["slower_nonconstant"], result.total, rtol=1e-10)
    np.testing.assert_allclose(result.groups["faster"], 0, atol=1e-20)


@pytest.mark.parametrize("failure", ["weights", "order", "shape", "range", "orthogonality"])
def test_rejects_invalid_or_underresolved_quadrature(failure: str) -> None:
    z, w = np.polynomial.legendre.leggauss(3)
    nodes = ((z + 1) / 2,)
    weights = (w / 2,)
    bounds = ((0.0, 1.0),)
    error = np.ones((3, 1))
    reference = 0.1 * np.pi**2
    if failure == "weights":
        weights = (weights[0] * 2,)
    if failure == "order":
        nodes = (nodes[0][::-1],)
    if failure == "shape":
        error = np.ones((2, 1))
    if failure == "range":
        reference = 1e5
    if failure == "orthogonality":
        reference = 0.1 * (2 * np.pi) ** 2
    with pytest.raises(ValueError):
        project_neumann_error(nodes, weights, bounds, error, reference)
