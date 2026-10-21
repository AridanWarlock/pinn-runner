"""Воспроизводимый CPU-аудит аналитического эталона TASK-39.5 (без обучения)."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.quadratic_reference import (  # noqa: E402
    reference_fields,
    series_coefficients,
    truncation_bound,
    truncation_bounds,
)

OUT = Path(__file__).resolve().parent
x = np.linspace(0, 1, 101)[:, None]
t = np.linspace(0, 1, 101)[None, :]
rows = []
coefficient_rows = []
balance_nodes, balance_weights = leggauss(128)
balance_x = ((balance_nodes + 1) / 2)[:, None]
balance_weights = balance_weights / 2
balance_times = np.array([0.0, 0.01, 0.05, 0.2, 0.5, 1.0])[None, :]
for b in np.linspace(1, 2, 5):
    coefficients = np.asarray(series_coefficients(float(b), 128).values)
    nu = np.arange(1, 129) * np.pi / np.log1p(1 / b)
    quadratures = []
    for count in [256, 512]:
        roots, weights = leggauss(count)
        nodes, weights = (roots + 1) / 2, weights / 2
        phi = (1 + nodes[:, None] / b) ** -0.5 * np.sin(np.log1p(nodes[:, None] / b) * nu)
        quadratures.append(
            2 / (b * np.log1p(1 / b)) * ((weights * np.sin(np.pi * nodes) ** 3) @ phi)
        )
    coefficient_rows.append(
        {
            "b": float(b),
            "modes": 128,
            "coordinate_quadrature_max_difference": float(
                np.max(abs(coefficients - quadratures[1]))
            ),
            "gauss_refinement_max_difference": float(np.max(abs(quadratures[0] - quadratures[1]))),
            "sum_estimated_quad_errors": sum(series_coefficients(float(b), 128).estimated_errors),
        }
    )
    for a in np.linspace(0.05, 0.15, 5):
        fields = reference_fields(x, t, a=float(a), b=float(b))
        refined = reference_fields(x, t, a=float(a), b=float(b), modes=128)
        k = a * (1 + x / b) ** 2
        kx = 2 * a / b * (1 + x / b)
        balance_fields = reference_fields(balance_x, balance_times, a=float(a), b=float(b))
        faces = reference_fields(np.array([[0.0], [1.0]]), balance_times, a=float(a), b=float(b))
        boundary_change = a * (1 + 1 / b) ** 2 * faces.dx[1] - a * faces.dx[0]
        balance_error = float(np.max(abs(balance_weights @ balance_fields.dt - boundary_change)))
        rows.append(
            {
                "a": float(a),
                "b": float(b),
                "field_cutoff64_128_max_difference": float(
                    np.max(abs(fields.temperature - refined.temperature))
                ),
                "dx_cutoff64_128_max_difference": float(np.max(abs(fields.dx - refined.dx))),
                "dxx_cutoff64_128_max_difference": float(np.max(abs(fields.dxx - refined.dxx))),
                "dt_cutoff64_128_max_difference": float(np.max(abs(fields.dt - refined.dt))),
                "pde_max": float(np.max(abs(fields.dt - k * fields.dxx - kx * fields.dx))),
                "heat_balance_max_error": balance_error,
                "missing_kx_pde_rms": float(np.sqrt(np.mean((fields.dt - k * fields.dxx) ** 2))),
            }
        )
for suffix, entries in [("cases", rows), ("coefficients", coefficient_rows)]:
    with (OUT / f"2026-10-21-quadratic-reference-{suffix}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
shape_response = []
for a in [0.05, 0.1, 0.15]:
    left = reference_fields(x, t, a=a, b=1.0).temperature
    right = reference_fields(x, t, a=a, b=2.0).temperature
    shape_response.append(
        {
            "a": a,
            "b1_vs_b2_relative_field_distance": float(
                np.linalg.norm(left - right) / np.linalg.norm(right)
            ),
        }
    )
summary = {
    "kind": "cpu_analytical_reference_audit_not_training",
    "hypotheses": len(rows),
    "grid": [101, 101],
    "modes": [64, 128],
    "positive_time_min": 0.01,
    "proven_field_tail_bound_at_time_min": float(truncation_bound(np.array(0.01))),
    "proven_derivative_tail_bounds_at_time_min": {
        key: float(getattr(truncation_bounds(np.array(0.01)), key)) for key in ["dx", "dxx", "dt"]
    },
    "quadrature_errors_are_estimates_not_certificates": True,
    "maxima": {key: max(row[key] for row in rows) for key in list(rows[0])[2:]},
    "coefficient_maxima": {
        key: max(row[key] for row in coefficient_rows) for key in list(coefficient_rows[0])[2:]
    },
    "shape_response": shape_response,
}
(OUT / "2026-10-21-quadratic-reference-audit.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
