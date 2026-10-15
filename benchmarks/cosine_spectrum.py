"""Discrete Neumann cosine modes on a complete uniform Cartesian prediction grid."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .cases import Array


@dataclass(frozen=True)
class CosineSpectrum:
    indices: tuple[tuple[int, ...], ...]
    eigenvalues: Array
    coefficients: Array
    modal_energy: Array
    sampled_error_energy: Array
    parseval_max_error: float

    def grouped_energy(self, reference_eigenvalue: float) -> dict[str, Array]:
        if not np.isfinite(reference_eigenvalue) or reference_eigenvalue <= 0:
            raise ValueError("Positive reference eigenvalue required")
        equal = np.isclose(self.eigenvalues, reference_eigenvalue, rtol=1e-12, atol=0)
        zero = np.array([not any(index) for index in self.indices])
        masks = {
            "constant": zero,
            "slower_nonconstant": (~zero) & (~equal) & (self.eigenvalues < reference_eigenvalue),
            "same_rate": equal,
            "faster": (~equal) & (self.eigenvalues > reference_eigenvalue),
        }
        return {name: self.modal_energy[mask].sum(axis=0) for name, mask in masks.items()}


def cosine_spectrum(
    spatial: dict[str, Array], error: Array, conductivity: float = 0.1
) -> CosineSpectrum:
    """Project sampled error, not solution, with tensor-product trapezoidal weights.

    Full discrete cosine basis is orthogonal for these weights. This is not a
    claim about the unresolved continuous error or non-Neumann eigenfunctions.
    """
    if not 1 <= len(spatial) <= 3 or conductivity <= 0 or not np.isfinite(conductivity):
        raise ValueError("One to three Cartesian axes and positive conductivity required")
    if error.ndim != 2 or not np.isfinite(error).all() or error.shape[1] < 1:
        raise ValueError("Error must be a finite space-by-time matrix")
    axes = []
    for column in spatial.values():
        if column.shape != (len(error), 1) or not np.isfinite(column).all():
            raise ValueError("Spatial coordinates must be aligned finite columns")
        nodes = np.unique(column[:, 0])
        if len(nodes) < 2 or not np.allclose(
            np.diff(nodes), nodes[1] - nodes[0], rtol=1e-6, atol=1e-12
        ):
            raise ValueError("Uniform axes including both endpoints required")
        axes.append(nodes)
    shape = tuple(len(axis) for axis in axes)
    if int(np.prod(shape)) != len(error):
        raise ValueError("A complete Cartesian grid without duplicate rows required")
    mesh = np.meshgrid(*axes, indexing="ij")
    if any(
        not np.array_equal(column[:, 0], expected.reshape(-1))
        for column, expected in zip(spatial.values(), mesh)
    ):
        raise ValueError("Spatial rows must follow Cartesian ij order")
    transformed = error.reshape((*shape, error.shape[1]))
    weights = np.ones(shape, dtype=np.float64)
    mode_norms = np.ones(shape, dtype=np.float64)
    lengths = []
    for index, nodes in enumerate(axes):
        length = float(nodes[-1] - nodes[0])
        lengths.append(length)
        counts = len(nodes) - 1
        quadrature = np.full(len(nodes), length / counts, dtype=np.float64)
        quadrature[[0, -1]] *= 0.5
        basis = np.cos(np.pi * np.outer(np.arange(len(nodes)), np.arange(len(nodes))) / counts)
        norms = np.full(len(nodes), length / 2, dtype=np.float64)
        norms[[0, -1]] = length  # The last resolvable mode has the endpoint norm.
        projection = basis * quadrature[None, :] / norms[:, None]
        transformed = np.moveaxis(np.tensordot(projection, transformed, axes=(1, index)), 0, index)
        broadcast = [1] * len(shape)
        broadcast[index] = len(nodes)
        weights *= quadrature.reshape(broadcast)
        mode_norms *= norms.reshape(broadcast)
    energy = transformed**2 * mode_norms[..., None]
    sampled_energy = (error.reshape((*shape, error.shape[1])) ** 2 * weights[..., None]).sum(
        axis=tuple(range(len(shape)))
    )
    reconstructed_energy = energy.sum(axis=tuple(range(len(shape))))
    if not np.allclose(reconstructed_energy, sampled_energy, rtol=1e-11, atol=1e-14):
        raise ArithmeticError("Discrete cosine energy does not reconstruct sampled error")
    indices = tuple(itertools.product(*(range(count) for count in shape)))
    rates = np.array(
        [
            conductivity * np.pi**2 * sum((k / length) ** 2 for k, length in zip(mode, lengths))
            for mode in indices
        ],
        dtype=np.float64,
    )
    return CosineSpectrum(
        indices,
        rates,
        transformed.reshape(-1, error.shape[1]),
        energy.reshape(-1, error.shape[1]),
        sampled_energy,
        float(np.max(np.abs(reconstructed_energy - sampled_energy))),
    )
