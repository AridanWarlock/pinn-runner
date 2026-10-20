"""Check the discrete spectral rate against a continuous piecewise-linear trial."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import roots_legendre

SOURCE = Path(__file__).with_name("2026-10-20-unforced-burgers-reference-audit.py")
SPEC = importlib.util.spec_from_file_location("burgers_reference", SOURCE)
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)
AUDIT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
REFERENCE_SHA = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
BASE = Path(__file__).with_name("2026-10-20-unforced-burgers-rayleigh")


def slope(x, t, nu):
    flat = x.reshape(-1)
    result = np.empty_like(flat)
    for start in range(0, flat.size, 256):
        result[start : start + 256] = REFERENCE.reference(
            flat[start : start + 256], t, nu, derivative=True
        )
    return result.reshape(x.shape)


def check(nu, t, nx, order):
    x = np.linspace(-1, 1, nx)
    h = x[1] - x[0]
    eigenvalues, eigenvectors = eigh_tridiagonal(
        -2 * nu / h**2 - slope(x[1:-1], t, nu) / 2,
        np.full(nx - 3, nu / h**2),
        select="i",
        select_range=(nx - 3, nx - 3),
    )
    z = np.r_[0.0, eigenvectors[:, 0], 0.0]
    # Continuous H1_0 trial: linear on each cell, exactly zero at both ends.
    norm_squared = h / 3 * np.sum(z[:-1] ** 2 + z[:-1] * z[1:] + z[1:] ** 2)
    diffusion = nu / h * np.sum(np.diff(z) ** 2)
    nodes, weights = roots_legendre(order)
    local = (nodes + 1) / 2
    points = x[:-1, None] + h * local
    trial = z[:-1, None] * (1 - local) + z[1:, None] * local
    potential = h / 2 * np.sum(slope(points, t, nu) * trial**2 * weights)
    rayleigh = (-diffusion - potential / 2) / norm_squared
    return dict(
        nu=nu,
        t=t,
        nx=nx,
        quadrature_per_cell=order,
        finite_difference_eigenvalue=float(eigenvalues[0]),
        continuous_trial_quotient=float(rayleigh),
        difference=float(eigenvalues[0] - rayleigh),
    )


rows = [
    check(nu, t, nx, order)
    for nu, peak in ((0.1, 0.2375), (0.01, 0.5375))
    for t in (0, peak, 1)
    for nx in (401, 801, 1601)
    for order in (4, 8)
]
quadrature_difference = max(
    abs(rows[i]["continuous_trial_quotient"] - rows[i + 1]["continuous_trial_quotient"])
    for i in range(0, len(rows), 2)
)
assert quadrature_difference < 1e-9
# Preserve the coarse-grid failure of the initial diagnostic tolerance.
# The continuous and difference quotients should converge, not agree at arbitrary h.
for row in rows:
    row["within_initial_tolerance"] = abs(row["difference"]) < 0.025
for nu, peak in ((0.1, 0.2375), (0.01, 0.5375)):
    for t in (0, peak, 1):
        selected = [
            r for r in rows if r["nu"] == nu and r["t"] == t and r["quadrature_per_cell"] == 8
        ]
        errors = [abs(r["difference"]) for r in selected]
        assert errors[2] < errors[1] < errors[0]
        assert 3.5 < errors[0] / errors[1] < 4.5
        assert 3.5 < errors[1] / errors[2] < 4.5
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == AUDIT_SHA
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == REFERENCE_SHA
with BASE.with_suffix(".csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
summary = dict(
    networks=0,
    optimizer_steps=0,
    rows=len(rows),
    audit_sha256=AUDIT_SHA,
    reference_sha256=REFERENCE_SHA,
    quadrature_difference_max=quadrature_difference,
    initial_tolerance=0.025,
    all_grids_within_initial_tolerance=all(r["within_initial_tolerance"] for r in rows),
    failed_initial_tolerance=[r for r in rows if not r["within_initial_tolerance"]],
    refinement_checks_passed=True,
    finest=[r for r in rows if r["nx"] == 1601 and r["quadrature_per_cell"] == 8],
    scope="Continuous trial quotient is evaluated numerically; it is not a certified upper bound on the continuous largest eigenvalue.",
)
BASE.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
