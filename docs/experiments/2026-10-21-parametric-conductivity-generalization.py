"""A smooth parameter-localized error evades finite material slices."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "docs/experiments/2026-10-21-parametric-conductivity"
spec = importlib.util.spec_from_file_location("reference", f"{PREFIX}-reference.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def bump(x):
    result = np.zeros_like(x, dtype=float)
    inside = np.abs(x) < 1
    result[inside] = np.exp(1 - 1 / (1 - x[inside] ** 2))
    return result


def average_residual(epsilon, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x = (nodes + 1) / 2
    a = 0.123 + epsilon * nodes[:, None, None]
    b = 0.137 + epsilon * nodes[None, :, None]
    xx = x[None, None, :]
    c = module.coefficient(xx, a, b)
    cx = -np.pi * c * (a * np.sin(np.pi * xx) + 2 * b * np.sin(2 * np.pi * xx))
    p = xx * (1 - xx)
    k = 2 * c - cx * (1 - 2 * xx)
    time_integral = p * p + p * k + k * k / 3
    envelope = bump(nodes) ** 2 * weights
    integral = np.einsum("i,j,k,ijk", envelope, envelope, weights / 2, time_integral)
    return float(0.1 * epsilon * np.sqrt(integral / 1.96))


def main():
    fine = module.reference(1024, 0.123, 0.137)
    coarse = module.reference(512, 0.123, 0.137)
    x, u = fine[:2]
    final_error = (
        0.1 * np.sqrt(np.trapz((x * (1 - x)) ** 2, x)) / np.sqrt(np.trapz(u[:, -1] ** 2, x))
    )
    grid_difference = float(np.abs(u[::2] - coarse[1]).max())
    rows = []
    for epsilon in (0.05, 0.025, 0.0125):
        a = np.array([v[0] for v in module.PAIRS])
        b = np.array([v[1] for v in module.PAIRS])
        checked_amplitude = 0.1 * bump((a - 0.123) / epsilon) * bump((b - 0.137) / epsilon)
        r32, r64 = [average_residual(epsilon, n) for n in (32, 64)]
        row = dict(
            epsilon=epsilon,
            registered_pairs=len(a),
            maximum_error_amplitude_on_registered_pairs=float(checked_amplitude.max()),
            parameter_space_average_pde_rms=r64,
            independent_quadrature_difference=abs(r32 - r64),
            unobserved_pair_final_relative_l2=float(final_error),
            unobserved_reference_grid_difference=grid_difference,
            initial_boundary_error=0.0,
            passed=bool(
                checked_amplitude.max() <= 1e-15
                and abs(r32 - r64) <= 1e-8
                and r64 < 0.003
                and final_error > 0.01
                and grid_difference <= 2e-5
            ),
        )
        rows.append(row)
    with Path(f"{PREFIX}-generalization.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        protocol_sha256=hashlib.sha256(
            (
                ROOT / "docs/parametric-conductivity-generalization-protocol-2026-10-21.md"
            ).read_bytes()
        ).hexdigest(),
        optimizer_steps=0,
        central_pair=[0.123, 0.137],
        applies_to="smooth constructed fields; not a trained finite network",
        all_passed=all(r["passed"] for r in rows),
    )
    Path(f"{PREFIX}-generalization.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2))
    print(rows)
    assert result["all_passed"], "Preserve outcomes without changing tolerances"


if __name__ == "__main__":
    main()
