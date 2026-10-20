"""Independent adaptive integrals and finite differences for the new reference."""

import numpy as np
import pytest
from scipy.integrate import quad

from benchmarks.burgers import reference
from benchmarks.cases import CASES
from benchmarks.catalog import UNFORCED_VISCOSITIES


@pytest.mark.parametrize("name,nu", list(UNFORCED_VISCOSITIES.items()))
def test_reference_independent_kernel_and_physical_domain(name: str, nu: float) -> None:
    config = CASES[name].config("cpu", 1, 11, True)
    assert config["domain"]["space"]["x"] == dict(interval=[-1, 1], points=801)
    assert config["domain"]["time"] == dict(interval=[0, 1], points=201)
    for x, t in ((0.0, 0.5375), (0.01, 1.0), (0.3, 0.3), (1.0, 0.5)):
        scale = np.sqrt(4 * nu * t)
        a = 1 / (2 * np.pi * nu)

        def density(z):
            return np.exp(-z * z - a * np.cos(np.pi * (x + scale * z)) - a)

        def integral(f):
            return sum(
                quad(lambda z: f(z) * density(z), k, k + 1, epsabs=1e-24, epsrel=1e-12)[0]
                for k in range(-12, 12)
            )

        total = integral(lambda z: 1.0)
        mean = integral(lambda z: z) / total
        variance = integral(lambda z: (z - mean) ** 2) / total
        observed = float(reference(np.array(x), np.array(t), nu))
        slope = float(reference(np.array(x), np.array(t), nu, derivative=True))
        assert observed == pytest.approx(-2 * np.sqrt(nu / t) * mean, abs=2e-12)
        assert slope == pytest.approx((1 - 2 * variance) / t, abs=2e-10)
    x = np.linspace(-1, 1, 101)
    np.testing.assert_allclose(reference(x, np.zeros_like(x), nu), -np.sin(np.pi * x), atol=3e-15)
    defects = []
    for step in (0.001, 0.0005):
        x, t = 0.01, 0.5375

        def f(x, t):
            return float(reference(np.array(x), np.array(t), nu))

        ux = (
            f(x - 2 * step, t) - 8 * f(x - step, t) + 8 * f(x + step, t) - f(x + 2 * step, t)
        ) / (12 * step)
        uxx = (
            -f(x + 2 * step, t)
            + 16 * f(x + step, t)
            - 30 * f(x, t)
            + 16 * f(x - step, t)
            - f(x - 2 * step, t)
        ) / (12 * step**2)
        ut = (
            f(x, t - 2 * step) - 8 * f(x, t - step) + 8 * f(x, t + step) - f(x, t + 2 * step)
        ) / (12 * step)
        defects.append(abs(ut + f(x, t) * ux - nu * uxx))
    assert defects[-1] < 5e-6
    if defects[0] > 1e-7:
        assert 12 < defects[0] / defects[1] < 20
