"""Независимые проверки аналитического эталона меняющейся формы k(x)."""

import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import leggauss

from benchmarks.quadratic_reference import reference_fields, series_coefficients, truncation_bound


@pytest.mark.parametrize("b", [1.0, 1.5, 2.0])
def test_coefficients_in_physical_coordinates(b):
    # Независимая квадратура по x; исходная интегрирует по log(1+x/b).
    roots, weights = leggauss(512)
    x, weights = (roots + 1) / 2, weights / 2
    length = np.log1p(1 / b)
    nu = np.arange(1, 65) * np.pi / length
    phi = (1 + x[:, None] / b) ** -0.5 * np.sin(np.log1p(x[:, None] / b) * nu)
    physical = 2 / (b * length) * ((weights * np.sin(np.pi * x) ** 3) @ phi)
    coefficients = series_coefficients(b)
    np.testing.assert_allclose(physical, coefficients.values, atol=2e-14, rtol=0)
    assert sum(coefficients.estimated_errors) < 1e-10


@pytest.mark.parametrize("a,b", [(0.05, 1.0), (0.1, 1.5), (0.15, 2.0)])
def test_series_physics_refinement_and_initial_data(a, b):
    x = np.linspace(0, 1, 101)[:, None]
    t = np.linspace(0, 1, 51)[None, :]
    fields = reference_fields(x, t, a=a, b=b)
    refined = reference_fields(x, t, a=a, b=b, modes=128)
    np.testing.assert_allclose(fields.temperature, refined.temperature, atol=2e-13, rtol=0)
    np.testing.assert_allclose(fields.temperature[:, 0], np.sin(np.pi * x[:, 0]) ** 3, atol=1e-15)
    assert np.max(np.abs(fields.temperature[[0, -1]])) == 0
    assert fields.temperature.min() >= -1e-14
    assert fields.temperature.max() <= 1 + 1e-14
    k = a * (1 + x / b) ** 2
    kx = 2 * a / b * (1 + x / b)
    np.testing.assert_allclose(fields.dt, k * fields.dxx + kx * fields.dx, atol=2e-13, rtol=0)
    assert np.max(np.abs(fields.dt - k * fields.dxx)) > 0.01
    assert float(truncation_bound(np.array(0.02))) < 1e-25


@pytest.mark.parametrize("a,b,n", [(0.05, 1.0, 1), (0.1, 1.5, 3), (0.15, 2.0, 7)])
def test_mode_eigenvalue_with_physical_autodiff(a, b, n):
    x = torch.linspace(0.01, 0.99, 41, dtype=torch.float64, requires_grad=True)
    length = np.log1p(1 / b)
    nu = n * np.pi / length
    phi = (1 + x / b) ** -0.5 * torch.sin(nu * torch.log1p(x / b))
    dx = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
    divergence = torch.autograd.grad((a * (1 + x / b) ** 2 * dx).sum(), x)[0]
    decay = a / b**2 * (nu**2 + 0.25)
    torch.testing.assert_close(divergence, -decay * phi, atol=2e-12, rtol=2e-13)


@pytest.mark.parametrize("a,b", [(0.05, 1.0), (0.1, 1.5), (0.15, 2.0)])
def test_derivatives_with_finite_differences(a, b):
    x = np.array([0.13, 0.37, 0.61, 0.89])
    t = np.array([0.11, 0.29, 0.53, 0.83])
    fields = reference_fields(x, t, a=a, b=b)
    h = 1e-5
    xp = reference_fields(x + h, t, a=a, b=b).temperature
    xm = reference_fields(x - h, t, a=a, b=b).temperature
    tp = reference_fields(x, t + h, a=a, b=b).temperature
    tm = reference_fields(x, t - h, a=a, b=b).temperature
    np.testing.assert_allclose((xp - xm) / (2 * h), fields.dx, atol=2e-8, rtol=0)
    np.testing.assert_allclose((tp - tm) / (2 * h), fields.dt, atol=2e-8, rtol=0)
    h2 = 2e-4
    xp2 = reference_fields(x + h2, t, a=a, b=b).temperature
    xm2 = reference_fields(x - h2, t, a=a, b=b).temperature
    np.testing.assert_allclose(
        (xp2 - 2 * fields.temperature + xm2) / h2**2, fields.dxx, atol=2e-5, rtol=0
    )


def test_wrong_operator_breaks_initial_heat_balance():
    roots, weights = leggauss(128)
    x, weights = (roots + 1) / 2, weights / 2
    a, b = 0.1, 1.5
    fields = reference_fields(x, np.zeros_like(x), a=a, b=b)
    wrong = float(weights @ (a * (1 + x / b) ** 2 * fields.dxx))
    correct = float(weights @ fields.dt)
    assert abs(correct) < 1e-13
    assert wrong == pytest.approx(8 * a / (3 * np.pi * b**2), abs=1e-13)


@pytest.mark.parametrize("a,b,modes", [(0.01, 1, 64), (0.1, 3, 64), (0.1, 1, 0)])
def test_reference_rejects_outside_contract(a, b, modes):
    with pytest.raises(ValueError):
        reference_fields(np.array([0.5]), np.array([0.1]), a=a, b=b, modes=modes)


@pytest.mark.parametrize("b", [1.0, 1.5, 2.0])
def test_fourth_derivative_used_in_tail_proof(b):
    s = torch.linspace(
        0.01, float(np.log1p(1 / b)) - 0.01, 31, dtype=torch.float64, requires_grad=True
    )
    x = b * torch.expm1(s)
    f = torch.sin(torch.pi * x) ** 3
    g = torch.exp(s / 2) * f
    derivative = g
    for _ in range(4):
        derivative = torch.autograd.grad(derivative.sum(), s, create_graph=True)[0]
    pi = torch.pi
    # Независимые производные f через первую/третью гармоники.
    f1 = (3 * pi * torch.cos(pi * x) - 3 * pi * torch.cos(3 * pi * x)) / 4
    f2 = (-3 * pi**2 * torch.sin(pi * x) + 9 * pi**2 * torch.sin(3 * pi * x)) / 4
    f3 = (-3 * pi**3 * torch.cos(pi * x) + 27 * pi**3 * torch.cos(3 * pi * x)) / 4
    f4 = (3 * pi**4 * torch.sin(pi * x) - 81 * pi**4 * torch.sin(3 * pi * x)) / 4
    q = x + b
    formula = torch.exp(s / 2) * (
        q**4 * f4 + 8 * q**3 * f3 + 14.5 * q**2 * f2 + 5 * q * f1 + f / 16
    )
    torch.testing.assert_close(derivative, formula, atol=2e-10, rtol=2e-12)


@pytest.mark.parametrize("modes", [16, 32, 64])
def test_derivative_tail_bounds_cover_observed_refinement(modes):
    from benchmarks.quadratic_reference import truncation_bounds

    x = np.linspace(0, 1, 71)[:, None]
    t = np.array([0.01, 0.02, 0.1, 0.7])[None, :]
    bound = truncation_bounds(t, modes)
    for a, b in [(0.05, 1.0), (0.1, 1.5), (0.15, 2.0)]:
        fields = reference_fields(x, t, a=a, b=b, modes=modes)
        refined = reference_fields(x, t, a=a, b=b, modes=128)
        for key in ["temperature", "dx", "dxx", "dt"]:
            difference = abs(getattr(fields, key) - getattr(refined, key))
            assert np.all(difference <= getattr(bound, key) + 1e-10), (key, a, b, modes)
