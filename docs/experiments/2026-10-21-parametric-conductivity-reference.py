"""CPU-only independent reference for a family of unforced diffusion problems."""

import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "docs/experiments/2026-10-21-parametric-conductivity"
TIMES = np.linspace(0, 1, 41)
PAIRS = [*product((-0.7, 0.0, 0.7), repeat=2)] + [
    (-0.43, 0.19),
    (0.17, -0.51),
    (0.49, 0.31),
    (-0.23, -0.37),
]
EIGENSOLVER_FALLBACKS = []


def coefficient(x, a, b, extra=0.0):
    return (
        np.exp(
            a * np.cos(np.pi * x) + b * np.cos(2 * np.pi * x) + extra * np.sin(2 * np.pi * x) ** 2
        )
        / 8
    )


def reference(n, a, b, extra=0.0, coefficient_function=None):
    x = np.linspace(0, 1, n + 1)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    samples = (x[:-1, None] + x[1:, None]) / 2 + nodes / (2 * n)
    values = (
        coefficient(samples, a, b, extra)
        if coefficient_function is None
        else coefficient_function(samples)
    )
    resistance = ((1 / values) @ weights) / (2 * n)
    g = 1 / resistance
    diagonal = -n * (g[:-1] + g[1:])
    off = n * g[1:-1]
    try:
        eigenvalues, vectors = eigh_tridiagonal(diagonal, off)
    except np.linalg.LinAlgError as error:
        eigenvalues, vectors = eigh_tridiagonal(diagonal, off, lapack_driver="stev")
        applied = diagonal[:, None] * vectors
        applied[:-1] += off[:, None] * vectors[1:]
        applied[1:] += off[:, None] * vectors[:-1]
        eigen_residual = float(
            np.abs(applied - vectors * eigenvalues).max()
            / max(np.abs(diagonal).max() + 2 * np.abs(off).max(), 1)
        )
        orthogonality = float(np.abs(vectors.T @ vectors - np.eye(n - 1)).max())
        EIGENSOLVER_FALLBACKS.append(
            dict(
                n=n,
                a=a,
                b=b,
                extra=extra,
                custom=coefficient_function is not None,
                error=str(error),
                fallback="stev",
                eigen_residual_relative=eigen_residual,
                orthogonality_max=orthogonality,
            )
        )
        assert eigen_residual <= 1e-10 and orthogonality <= 1e-10
    initial = np.sin(np.pi * x[1:-1])
    interior = vectors @ ((vectors.T @ initial)[:, None] * np.exp(eigenvalues[:, None] * TIMES))
    u = np.pad(interior, ((1, 1), (0, 0)))
    return x, u, g, diagonal, off, eigenvalues, vectors


def main():
    rows, slices = [], []
    # Baseline comparison is a negative control, never a replacement reference.
    xbase, base, *_ = reference(1024, 0, 0)
    exact = np.sin(np.pi * xbase[:, None]) * np.exp(-(np.pi**2) * TIMES / 8)
    baseline_error = float(np.max(np.abs(base - exact)))
    for index, (a, b) in enumerate(PAIRS):
        grids = {n: reference(n, a, b) for n in (128, 256, 512, 1024)}
        x, u, g, diagonal, off, eigenvalues, vectors = grids[1024]
        differences = {
            n: float(np.max(np.abs(grids[n][1] - grids[2 * n][1][::2]))) for n in (128, 256, 512)
        }
        energy = np.sum(u**2, axis=0) / 2048
        dudt = (
            diagonal[:, None] * u[1:-1]
            + np.pad(off[:, None] * u[2:-1], ((0, 1), (0, 0)))
            + np.pad(off[:, None] * u[1:-2], ((1, 0), (0, 0)))
        )
        energy_rate = np.sum(u[1:-1] * dudt, axis=0) / 1024
        dissipation = np.sum(g[:, None] * np.diff(u, axis=0) ** 2, axis=0)
        coarse = grids[128]
        matrix = diags((coarse[4], coarse[3], coarse[4]), (-1, 0, 1), format="csr")
        independent = expm_multiply(
            matrix, np.sin(np.pi * coarse[0][1:-1]), start=0, stop=1, num=41
        ).T
        exp_error = float(np.max(np.abs(independent - coarse[1][1:-1])))
        row = dict(
            a=a,
            b=b,
            c_min_bound=float(np.exp(-abs(a) - abs(b)) / 8),
            c_max_bound=float(np.exp(abs(a) + abs(b)) / 8),
            difference_128_256=differences[128],
            difference_256_512=differences[256],
            difference_512_1024=differences[512],
            refinement_ratio=differences[256] / differences[512],
            eigenvalue_max=float(eigenvalues.max()),
            energy_max_increase=float(np.diff(energy).max()),
            energy_balance_error=float(np.max(np.abs(energy_rate + dissipation))),
            independent_exponential_max_error=exp_error,
            initial_max_error=float(np.max(np.abs(u[:, 0] - np.sin(np.pi * x)))),
            boundary_max=float(np.max(np.abs(u[[0, -1]]))),
            minimum=float(u.min()),
            maximum=float(u.max()),
            constant_coefficient_model_max_error=float(np.max(np.abs(u - base))),
            final_l2=float(np.sqrt(np.trapz(u[:, -1] ** 2, x))),
            passed=False,
        )
        row["passed"] = bool(
            differences[512] <= 2e-5
            and eigenvalues.max() < 0
            and row["energy_max_increase"] <= 1e-12
            and row["energy_balance_error"] <= 1e-9
            and exp_error <= 1e-10
            and row["initial_max_error"] <= 1e-12
            and row["boundary_max"] <= 1e-12
            and row["minimum"] >= -1e-11
            and row["maximum"] <= 1 + 1e-11
        )
        rows.append(row)
        for xi in (128, 256, 512, 768, 896):
            for ti in (0, 4, 10, 20, 40):
                slices.append(dict(a=a, b=b, x=float(x[xi]), t=float(TIMES[ti]), u=u[xi, ti]))
        print(index + 1, a, b, row["passed"], differences[512], flush=True)

    # Smooth coefficients agreeing on an open neighbourhood of x=1/4.
    def remote_bump(x):
        if x <= 0.5 or x >= 1:
            return 0.0
        z = (x - 0.75) / 0.25
        return np.exp(1 - 1 / (1 - z * z))

    def remote_c(x):
        return np.exp(remote_bump(x)) / 8

    total_resistance, quad_error = quad(
        lambda x: 1 / remote_c(x), 0, 1, points=[0.5, 0.75], epsabs=1e-12, epsrel=1e-12
    )
    local_x = 0.25
    remote_u = 8 * local_x / total_resistance
    nonlocality = dict(
        x=local_x,
        constant_c=0.125,
        remote_c_at_query=remote_c(local_x),
        common_neighbourhood=[0, 0.5],
        constant_solution=local_x,
        remote_solution=remote_u,
        solution_gap=remote_u - local_x,
        unavoidable_worst_case_absolute_error=(remote_u - local_x) / 2,
        quadrature_error_estimate=quad_error,
        passed=bool(remote_u - local_x > 0.01 and remote_c(local_x) == 0.125),
    )
    for suffix, data in (("reference.csv", rows), ("reference-slices.csv", slices)):
        with Path(f"{PREFIX}-{suffix}").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    protocol = ROOT / "docs/parametric-conductivity-protocol-2026-10-21.md"
    summary = dict(
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        training_steps=0,
        eigensolver_fallbacks=EIGENSOLVER_FALLBACKS,
        parameter_pairs=len(rows),
        baseline_analytic_max_error=baseline_error,
        nonlocality=nonlocality,
        all_passed=bool(
            all(r["passed"] for r in rows) and baseline_error <= 1e-6 and nonlocality["passed"]
        ),
    )
    Path(f"{PREFIX}-reference.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))
    assert summary["all_passed"], "Reference controls failed; preserve results and investigate"


if __name__ == "__main__":
    main()
