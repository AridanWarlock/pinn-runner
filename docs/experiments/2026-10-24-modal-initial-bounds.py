"""Независимые начальные нормы на новых формах и пол модальной поправки M=32."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.legendre import leggauss

OUT = Path(__file__).resolve().parent
weights_path = OUT / "2026-10-24-modal-control-weights.csv"
raw = np.loadtxt(weights_path, delimiter=",", skiprows=1)
assert np.array_equal(raw[:, 0], np.arange(1, 65))
coefficients = raw[:, 1:].T
A = 0.15
CORRECTION_MODES = 32  # Одна заранее заданная проверка пола, без подбора M.


def audit(b, nodes):
    roots, w = leggauss(nodes)
    xi, w = (roots + 1) / 2, w / 2
    length = np.log1p(1 / b)
    r = np.exp(length * xi)
    x = b * np.expm1(length * xi)
    measure = b * length * r * w
    n = np.arange(1, 65)
    nu = n * np.pi / length
    sine = np.sin(np.pi * xi[:, None] * n[None, :])
    cosine = np.cos(np.pi * xi[:, None] * n[None, :])
    phi = r[:, None] ** -0.5 * sine
    dx = r[:, None] ** -1.5 / b * (nu * cosine - 0.5 * sine)
    f = np.sin(np.pi * x) ** 3
    fx = 3 * np.pi * np.sin(np.pi * x) ** 2 * np.cos(np.pi * x)
    fxx = (
        6 * np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * x) ** 2
        - 3 * np.pi**2 * np.sin(np.pi * x) ** 3
    )
    fitted = chebval(2 * b - 3, coefficients)
    assert abs(measure @ f**2 - 5 / 16) < 1e-12
    error = np.sqrt(measure @ (phi @ fitted - f) ** 2)
    energy_error = np.sqrt(measure @ (A * r**2 * (dx @ fitted - fx) ** 2))
    projected = 2 * ((w * np.sqrt(r) * f) @ sine)
    c1 = projected[0] * np.sqrt(b * length / 2)
    eigen = A / b**2 * (nu**2 + 0.25)
    # Точное IC-представление: f*exp(-lambda1*t) + sum d_n phi_n (exp(-lambda_n*t)-exp(-lambda1*t)).
    # При d_n=c_n известный пространственный остаток содержит только отброшенную часть.
    operator_f = A * r**2 * fxx + 2 * A / b * r * fx
    spatial_residual = -(operator_f + eigen[0] * f) + phi[:, 1:CORRECTION_MODES] @ (
        (eigen[0] - eigen[1:CORRECTION_MODES]) * projected[1:CORRECTION_MODES]
    )
    temporal_norm = np.sqrt(-np.expm1(-2 * eigen[0]) / (2 * eigen[0]))
    floor = np.sqrt(measure @ spatial_residual**2) * temporal_norm
    return dict(
        b=b,
        initial_l2=error,
        initial_energy_error=energy_error,
        first_projection=c1,
        uniform_time_field_bound=error / c1,
        uniform_time_dx_flux_bound=(1 + 1 / b) * energy_error / (np.sqrt(eigen[0]) * c1),
        hard_initial_modal_pde_floor=floor,
    )


rows = []
for b in 1 + (np.arange(128) + 0.5) / 128:
    primary, independent = audit(float(b), 512), audit(float(b), 1024)
    primary["quadrature_norm_disagreement"] = max(
        abs(primary[k] - independent[k])
        for k in ("initial_l2", "initial_energy_error", "hard_initial_modal_pde_floor")
    )
    assert primary["quadrature_norm_disagreement"] < 1e-10
    rows.append(primary)
# Нормировка и энергия базиса независимо проверяются на случайной конечной сумме.
roots, w = leggauss(512)
xi = (roots + 1) / 2
w = w / 2
b = 1.37
length = np.log1p(1 / b)
r = np.exp(length * xi)
n = np.arange(1, 65)
nu = n * np.pi / length
phi = r[:, None] ** -0.5 * np.sin(np.pi * xi[:, None] * n)
dx = (
    r[:, None] ** -1.5
    / b
    * (nu * np.cos(np.pi * xi[:, None] * n) - 0.5 * np.sin(np.pi * xi[:, None] * n))
)
measure = b * length * r * w
eigen = A / b**2 * (nu**2 + 0.25)
delta = np.random.default_rng(20261024).normal(size=64) * 0.01
identities = []
for time in (0, 0.2, 1):
    modes = delta * np.exp(-eigen * time)
    value = float(measure @ (phi @ modes) ** 2)
    expected = float(b * length / 2 * np.sum(modes**2))
    energy = float(measure @ (A * r**2 * (dx @ modes) ** 2))
    expected_energy = float(b * length / 2 * np.sum(eigen * modes**2))
    np.testing.assert_allclose(value, expected, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(energy, expected_energy, rtol=1e-11, atol=1e-15)
    identities.append(
        dict(
            time=time,
            l2_squared_error=abs(value - expected),
            energy_squared_error=abs(energy - expected_energy),
        )
    )
with Path(__file__).with_suffix(".csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
summary = dict(
    kind="initial_only_bounds_and_oracle_floor_not_neural_training",
    new_shapes=128,
    shape_grid="1+(j+0.5)/128",
    correction_modes=CORRECTION_MODES,
    weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
    quadrature_nodes=[512, 1024],
    maxima={
        key: max(row[key] for row in rows)
        for key in rows[0]
        if key not in ("b", "first_projection")
    },
    minimum_first_projection=min(row["first_projection"] for row in rows),
    finite_sum_norm_checks=identities,
    continuous_norms_certified=False,
)
Path(__file__).with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
