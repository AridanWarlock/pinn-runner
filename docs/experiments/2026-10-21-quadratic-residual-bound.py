"""Непрерывная энергетическая оценка; числа квадратуры не являются сертификатом."""

import csv
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.catalog import QUADRATIC_EVALUATION  # noqa: E402
from benchmarks.quadratic_reference import reference_fields  # noqa: E402

nodes, weights = leggauss(128)
nodes, weights = (nodes + 1) / 2, weights / 2
rows = []
for a, b in QUADRATIC_EVALUATION["hypotheses"]:
    length = np.log1p(1 / b)
    decay = a / b**2 * ((np.pi / length) ** 2 + 0.25)

    def integrand(x):
        psi = (1 + x / b) ** -0.5 * np.sin(np.pi * np.log1p(x / b) / length)
        psi /= np.sqrt(b * length / 2)
        return np.sin(np.pi * x) ** 3 * psi

    projection, estimate = quad(integrand, 0, 1, epsabs=1e-13, epsrel=1e-13)
    independent = float(weights @ integrand(nodes))
    assert abs(projection - independent) < 1e-13
    assert projection > 0
    factor = np.sqrt(np.expm1(2 * decay) / (2 * decay)) / projection
    # Первая мода с оптимальной временной невязкой достигает числителя оценки.
    final_norm = np.sqrt(
        weights @ reference_fields(nodes, np.ones_like(nodes), a=a, b=b).temperature ** 2
    )
    error_norm = 0.02 * np.sqrt(-np.expm1(-2 * decay) / (2 * decay))
    actual = float(error_norm / final_norm)
    assert actual <= 0.02 * factor * (1 + 1e-12)
    initial_dx = 3 * np.pi * np.sin(np.pi * nodes) ** 2 * np.cos(np.pi * nodes)
    rayleigh = float(weights @ (a * (1 + nodes / b) ** 2 * initial_dx**2)) / (5 / 16)
    poincare = a * np.pi**2
    projection_free = (
        0.03
        * np.sqrt(5 / 16)
        * np.exp(-rayleigh)
        / np.sqrt(-np.expm1(-2 * poincare) / (2 * poincare))
    )
    assert projection_free <= 0.03 / factor
    times = np.array([0.1, 0.25, 0.5, 1.0])
    exact = reference_fields(nodes[:, None], times[None, :], a=a, b=b).temperature
    norms = np.sqrt(weights @ exact**2)
    lower = np.sqrt(5 / 16) * np.exp(-rayleigh * times)
    assert np.all(norms >= lower - 1e-13)

    rows.append(
        dict(
            a=a,
            b=b,
            first_decay=decay,
            initial_projection=projection,
            quadrature_error_estimate=estimate,
            quadrature_disagreement=abs(projection - independent),
            sufficient_residual_l2_for_3pct=0.03 / factor,
            relative_bound_at_residual_002=0.02 * factor,
            first_mode_counterexample_relative_final=actual,
            bound_to_counterexample_ratio=0.02 * factor / actual,
            initial_rayleigh_quotient=rayleigh,
            poincare_decay_lower_bound=poincare,
            projection_free_sufficient_residual=projection_free,
            minimum_jensen_check_ratio=float(np.min(norms / lower)),
        )
    )
output = Path(__file__).with_suffix(".csv")
with output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(
    f"{len(rows)} cases; sufficient residual range: "
    f"{min(r['sufficient_residual_l2_for_3pct'] for r in rows):.9f} to "
    f"{max(r['sufficient_residual_l2_for_3pct'] for r in rows):.9f}"
)
