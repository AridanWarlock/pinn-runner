"""Аналитический ряд для T_t = (a(1+x/b)^2 T_x)_x, T(x,0)=sin(pi*x)^3.

Эталон оценки, не обучающие данные. Квадратурная оценка не является сертификатом.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SeriesCoefficients:
    values: tuple[float, ...]
    estimated_errors: tuple[float, ...]


def _validate(a: float, b: float, modes: int) -> None:
    if not np.isfinite(a) or not 0.05 <= a <= 0.15:
        raise ValueError("a must be in [0.05, 0.15]")
    if not np.isfinite(b) or not 1.0 <= b <= 2.0:
        raise ValueError("b must be in [1, 2]")
    if isinstance(modes, bool) or not isinstance(modes, int) or modes < 1:
        raise ValueError("modes must be a positive integer")


@lru_cache(maxsize=64)
def series_coefficients(b: float, modes: int = 64) -> SeriesCoefficients:
    """Коэффициенты синус-ряда в логарифмической координате."""
    _validate(0.1, b, modes)
    length = float(np.log1p(1.0 / b))

    def initial(s: float) -> float:
        return float(np.exp(s / 2) * np.sin(np.pi * b * np.expm1(s)) ** 3)

    values: list[float] = []
    errors: list[float] = []
    for n in range(1, modes + 1):
        result = quad(
            initial,
            0.0,
            length,
            weight="sin",
            wvar=n * np.pi / length,
            epsabs=1e-13,
            epsrel=1e-13,
            full_output=1,
        )
        if len(result) != 3:
            raise ArithmeticError(f"Coefficient quadrature did not converge: b={b}, n={n}")
        value, error, _ = result
        values.append(float(2 * value / length))
        errors.append(float(2 * error / length))
    return SeriesCoefficients(tuple(values), tuple(errors))


@dataclass(frozen=True)
class ReferenceFields:
    temperature: Array
    dx: Array
    dxx: Array
    dt: Array


def reference_fields(
    x: Array, t: Array, *, a: float, b: float, modes: int = 64
) -> ReferenceFields:
    """Физические производные ряда; при t=0 используются точные начальные данные."""
    _validate(a, b, modes)
    x, t = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(t, dtype=np.float64))
    if not np.all(np.isfinite(x)) or not np.all((x >= 0) & (x <= 1)):
        raise ValueError("x must be finite and in [0, 1]")
    if not np.all(np.isfinite(t)) or not np.all((t >= 0) & (t <= 1)):
        raise ValueError("t must be finite and in [0, 1]")
    r = 1 + x[..., None] / b
    s = np.log1p(x[..., None] / b)
    length = np.log1p(1.0 / b)
    nu = np.arange(1, modes + 1, dtype=np.float64) * np.pi / length
    decay = (a / b**2) * (nu**2 + 0.25)
    factors = np.asarray(series_coefficients(b, modes).values) * np.exp(-t[..., None] * decay)
    sine, cosine = np.sin(s * nu), np.cos(s * nu)
    phi = r**-0.5 * sine
    field = np.sum(factors * phi, axis=-1)
    dx = np.sum(factors * r**-1.5 / b * (nu * cosine - 0.5 * sine), axis=-1)
    dxx = np.sum(factors * r**-2.5 / b**2 * ((0.75 - nu**2) * sine - 2 * nu * cosine), axis=-1)
    dt = np.sum(-decay * factors * phi, axis=-1)
    initial = t == 0
    sn, cs = np.sin(np.pi * x), np.cos(np.pi * x)
    initial_dx = 3 * np.pi * sn**2 * cs
    initial_dxx = 6 * np.pi**2 * sn * cs**2 - 3 * np.pi**2 * sn**3
    dx = np.where(initial, initial_dx, dx)
    dxx = np.where(initial, initial_dxx, dxx)
    field = np.where(initial, sn**3, field)
    dt = np.where(initial, a * (1 + x / b) ** 2 * dxx + 2 * a / b * (1 + x / b) * dx, dt)
    boundary = (x == 0) | (x == 1)
    field = np.where(boundary, 0.0, field)
    dt = np.where(boundary, 0.0, dt)
    return ReferenceFields(field, dx, dxx, dt)


def truncation_bounds(t: Array, modes: int = 64) -> ReferenceFields:
    """Границы хвостов поля/производных; без квадратуры и округления."""
    _validate(0.1, 1.0, modes)
    times = np.asarray(t, dtype=np.float64)
    if not np.all(np.isfinite(times)) or not np.all((times >= 0) & (times <= 1)):
        raise ValueError("t must be finite and in [0, 1]")
    pi = np.pi
    derivative_bound = np.sqrt(2.0) * (
        81 * 21 * pi**4 + 216 * 7.5 * pi**3 + 14.5 * 9 * 3 * pi**2 + 15 * 1.5 * pi + 1 / 16
    )
    coefficient_bound = 2 * derivative_bound * np.log(2.0) ** 4 / pi**4
    minimum_decay = 0.05 * pi**2 / (2 * np.log(1.5)) ** 2
    exponential = np.exp(-minimum_decay * (modes + 1) ** 2 * times)
    frequency_bound = pi / np.log(1.5)
    return ReferenceFields(
        np.asarray(coefficient_bound / (3 * modes**3) * exponential),
        np.asarray(
            coefficient_bound
            * (frequency_bound / (2 * modes**2) + 1 / (6 * modes**3))
            * exponential
        ),
        np.asarray(
            coefficient_bound
            * (frequency_bound**2 / modes + frequency_bound / modes**2 + 0.25 / modes**3)
            * exponential
        ),
        np.asarray(
            coefficient_bound
            * 0.15
            * (frequency_bound**2 / modes + 1 / (12 * modes**3))
            * exponential
        ),
    )


def truncation_bound(t: Array, modes: int = 64) -> Array:
    """Граница хвоста только поля; совместимый краткий интерфейс."""
    return truncation_bounds(t, modes).temperature
