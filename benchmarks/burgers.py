"""Independent positive Cole-Hopf reference, never used as training data."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_hermite

Array = NDArray[np.float64]


@lru_cache
def _rule(order: int) -> tuple[Array, Array]:
    nodes, weights = roots_hermite(order)
    positive = weights > 0
    return nodes[positive], np.log(weights[positive])


def reference(
    x: Array, t: Array, viscosity: float, *, derivative: bool = False, order: int = 512
) -> Array:
    if viscosity not in (0.1, 0.01) or order < 16:
        raise ValueError("Registered viscosity and a positive quadrature order >=16 required")
    xx, tt = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(t, dtype=np.float64))
    if not np.isfinite(xx).all() or not np.isfinite(tt).all() or np.any(tt < 0):
        raise ValueError("Finite coordinates and nonnegative time required")
    nodes, log_weights = _rule(order)
    flat_x, flat_t = xx.reshape(-1), tt.reshape(-1)
    result = np.empty(xx.size, dtype=np.float64)
    for start in range(0, xx.size, 256):
        stop = start + 256
        y = flat_x[start:stop]
        y = y[:, None] + np.sqrt(4 * viscosity * flat_t[start:stop, None]) * nodes
        exponent = log_weights - np.cos(np.pi * y) / (2 * np.pi * viscosity)
        weights = np.exp(exponent - exponent.max(axis=1, keepdims=True))
        total = weights.sum(axis=1)
        sine = np.sin(np.pi * y)
        mean = (weights * sine).sum(axis=1) / total
        if derivative:
            variance = (weights * (sine - mean[:, None]) ** 2).sum(axis=1) / total
            result[start:stop] = -(weights * np.pi * np.cos(np.pi * y)).sum(
                axis=1
            ) / total - variance / (2 * viscosity)
        else:
            result[start:stop] = -mean
    return result.reshape(xx.shape)


@lru_cache(maxsize=8)
def grid_reference(
    x: tuple[float, ...], t: tuple[float, ...], viscosity: float, derivative: bool = False
) -> Array:
    result = reference(
        np.array(x)[:, None], np.array(t)[None, :], viscosity, derivative=derivative
    )
    result.setflags(write=False)
    return result
