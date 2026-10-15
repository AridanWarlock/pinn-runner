"""Discrete cosine Parseval identity and distinct slow error modes."""

import numpy as np
import pytest

from benchmarks.cosine_spectrum import cosine_spectrum


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_shifted_box_cosine_coefficients_and_exact_sampled_energy(dimensions: int) -> None:
    axes = [np.linspace(-0.4 + i * 0.3, 1.2 + i * 0.5, 5 + i) for i in range(dimensions)]
    mesh = np.meshgrid(*axes, indexing="ij")
    spatial = {name: c.reshape(-1, 1) for name, c in zip("xyz", mesh)}
    modes = [tuple(0 for _ in axes), tuple(1 for _ in axes), tuple(len(x) - 1 for x in axes)]
    amplitudes = np.array([[0.2, 0.3], [0.5, -0.4], [0.1, 0.2]])
    error = np.zeros((int(np.prod([len(x) for x in axes])), 2))
    for mode, amplitude in zip(modes, amplitudes):
        phi = np.ones((len(error), 1))
        for column, axis, k in zip(spatial.values(), axes, mode):
            phi *= np.cos(np.pi * k * (column - axis[0]) / (axis[-1] - axis[0]))
        error += phi * amplitude
    result = cosine_spectrum(spatial, error)
    for mode, amplitude in zip(modes, amplitudes):
        np.testing.assert_allclose(
            result.coefficients[result.indices.index(mode)], amplitude, rtol=1e-12, atol=1e-13
        )
    reference_rate = result.eigenvalues[result.indices.index(modes[1])]
    groups = result.grouped_energy(float(reference_rate))
    np.testing.assert_allclose(
        sum(groups.values()), result.sampled_error_energy, rtol=1e-12, atol=1e-13
    )
    assert result.parseval_max_error < 1e-12
    assert all(
        np.max(np.abs(result.coefficients[i])) < 1e-12
        for i, index in enumerate(result.indices)
        if index not in modes
    )


def test_slow_and_reference_rates_are_not_the_same_mode() -> None:
    axes = np.linspace(0, 1, 5)
    mesh = np.meshgrid(axes, axes, axes, indexing="ij")
    spatial = {name: c.reshape(-1, 1) for name, c in zip("xyz", mesh)}
    error = 0.2 * np.cos(np.pi * spatial["x"]) + 0 * spatial["y"] + 0 * spatial["z"]
    result = cosine_spectrum(spatial, error)
    groups = result.grouped_energy(0.1 * 3 * np.pi**2)
    np.testing.assert_allclose(
        groups["slower_nonconstant"], result.sampled_error_energy, atol=1e-14
    )
    np.testing.assert_allclose(groups["constant"], 0, atol=1e-14)
    np.testing.assert_allclose(groups["same_rate"], 0, atol=1e-14)


@pytest.mark.parametrize("invalid", ["nonuniform", "row_order", "duplicate", "shape", "nan"])
def test_rejects_incompatible_prediction_grid(invalid: str) -> None:
    nodes = np.array([0.0, 0.5, 1.0])
    spatial = {"x": nodes.reshape(-1, 1)}
    error = np.ones((3, 2))
    if invalid == "nonuniform":
        spatial["x"][1] = 0.3
    if invalid == "row_order":
        spatial["x"] = spatial["x"][::-1].copy()
    if invalid == "duplicate":
        spatial["x"][1] = 0
    if invalid == "shape":
        error = error.T
    if invalid == "nan":
        error[1, 0] = np.nan
    with pytest.raises(ValueError):
        cosine_spectrum(spatial, error)


def test_zero_field_has_no_spurious_modes_or_undefined_energies() -> None:
    nodes = np.linspace(0, 1, 5)
    result = cosine_spectrum({"x": nodes.reshape(-1, 1)}, np.zeros((5, 2)))
    assert np.array_equal(result.modal_energy, np.zeros_like(result.modal_energy))
    # Tiny conductivity must not classify all nonconstant rates as zero.
    result = cosine_spectrum({"x": nodes.reshape(-1, 1)}, np.ones((5, 1)), conductivity=1e-20)
    groups = result.grouped_energy(1e-20 * np.pi**2)
    np.testing.assert_allclose(groups["constant"], result.sampled_error_energy, atol=1e-14)
