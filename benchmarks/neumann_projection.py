"""Low heat eigenmodes and their orthogonal remainder on an independent quadrature."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .cases import Array


@dataclass(frozen=True)
class NeumannEnergy:
    indices: tuple[tuple[int, ...], ...]
    rates: Array
    coefficients: Array
    groups: dict[str, Array]
    total: Array
    gram_defect: float
    reconstruction_max_error: float
    normalization_defect: float


def project_neumann_error(
    nodes: tuple[Array, ...],
    weights: tuple[Array, ...],
    bounds: tuple[tuple[float, float], ...],
    error: Array,
    reference_rate: float,
    conductivity: float = 0.1,
) -> NeumannEnergy:
    """Resolve every eigenmode no faster than reference, leaving a fast remainder.

    Near-orthogonality is checked in the supplied positive quadrature. This is
    a numerical projection, not a certified bound on unsampled error.
    """
    if not 1 <= len(nodes) <= 3 or len(nodes) != len(weights) or len(nodes) != len(bounds):
        raise ValueError("Matching one to three quadrature axes required")
    if (
        not np.isfinite([reference_rate, conductivity]).all()
        or min(reference_rate, conductivity) <= 0
    ):
        raise ValueError("Positive finite reference rate and conductivity required")
    lengths = []
    for axis, rule, (lower, upper) in zip(nodes, weights, bounds):
        if not np.isfinite([lower, upper]).all() or upper <= lower:
            raise ValueError("Finite increasing bounds required")
        if (
            axis.ndim != 1
            or len(axis) < 2
            or rule.shape != axis.shape
            or not np.isfinite(axis).all()
            or not np.isfinite(rule).all()
            or np.any(np.diff(axis) <= 0)
            or np.any(rule <= 0)
        ):
            raise ValueError("Increasing finite nodes with matching positive weights required")
        if (
            axis[0] < lower
            or axis[-1] > upper
            or not np.isclose(rule.sum(), upper - lower, rtol=1e-10, atol=1e-12)
        ):
            raise ValueError("Quadrature must integrate the stated interval")
        lengths.append(upper - lower)
    shape = tuple(len(axis) for axis in nodes)
    if (
        error.ndim != 2
        or error.shape[0] != int(np.prod(shape))
        or error.shape[1] < 1
        or not np.isfinite(error).all()
    ):
        raise ValueError("Finite Cartesian space-by-time error required")
    maxima = [
        int(np.floor(np.sqrt(reference_rate / conductivity) * length / np.pi + 1e-12))
        for length in lengths
    ]
    if any(maximum >= len(axis) for maximum, axis in zip(maxima, nodes)):
        raise ValueError("Quadrature has fewer nodes than the required eigenmode range")
    indices = tuple(
        index
        for index in itertools.product(*(range(k + 1) for k in maxima))
        if conductivity * np.pi**2 * sum((k / length) ** 2 for k, length in zip(index, lengths))
        <= reference_rate * (1 + 1e-12)
    )
    mesh = np.meshgrid(*nodes, indexing="ij")
    weight_mesh = np.meshgrid(*weights, indexing="ij")
    volume_weights = np.prod(np.stack(weight_mesh), axis=0).reshape(-1)
    basis = []
    for index in indices:
        value = np.ones(shape, dtype=np.float64)
        for column, k, (lower, upper) in zip(mesh, index, bounds):
            value *= np.cos(k * np.pi * (column - lower) / (upper - lower))
        basis.append(value.reshape(-1))
    phi = np.stack(basis, axis=1)
    gram = phi.T @ (volume_weights[:, None] * phi)
    diagonal = np.diag(gram)
    if np.any(diagonal <= np.finfo(float).tiny):
        raise ValueError("Quadrature does not resolve the selected eigenmodes")
    normalized_gram = gram / np.sqrt(diagonal[:, None] * diagonal[None, :])
    defect = float(np.max(np.abs(normalized_gram - np.eye(len(indices)))))
    if defect > 1e-6:
        raise ValueError("Quadrature does not resolve orthogonal eigenmodes; increase points")
    coefficients = (phi.T @ (volume_weights[:, None] * error)) / diagonal[:, None]
    energies = coefficients**2 * diagonal[:, None]
    rates = np.array(
        [
            conductivity * np.pi**2 * sum((k / length) ** 2 for k, length in zip(index, lengths))
            for index in indices
        ],
        dtype=np.float64,
    )
    constant = np.array([not any(index) for index in indices])
    equal = np.isclose(rates, reference_rate, rtol=1e-12, atol=0)
    remainder = error - phi @ coefficients
    total = (volume_weights[:, None] * error**2).sum(axis=0)
    groups = {
        "constant": energies[constant].sum(axis=0),
        "slower_nonconstant": energies[(~constant) & (~equal)].sum(axis=0),
        "same_rate": energies[equal].sum(axis=0),
        "faster": (volume_weights[:, None] * remainder**2).sum(axis=0),
    }
    exact_norms = np.array(
        [np.prod(lengths) / 2 ** sum(k > 0 for k in index) for index in indices],
        dtype=np.float64,
    )
    return NeumannEnergy(
        indices=indices,
        rates=rates,
        coefficients=coefficients,
        groups=groups,
        total=total,
        gram_defect=defect,
        reconstruction_max_error=float(np.max(np.abs(sum(groups.values()) - total))),
        normalization_defect=float(np.max(np.abs(diagonal / exact_norms - 1))),
    )
