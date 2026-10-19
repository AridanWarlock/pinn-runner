"""CPU refinement study: coefficient interpolation, strong residual and heat balance."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinnstorch.api import Coordinates, Equation  # noqa: E402
from pinnstorch.coefficients import RegularGridCoefficient  # noqa: E402
from pinnstorch.runner.coefficients import CoefficientBank  # noqa: E402

K0 = 1 / 8
TIME_INTEGRAL = -math.expm1(-2) / 2


def measure(cells: int) -> dict[str, float | int]:
    nodes = torch.linspace(0, 1, cells + 1, dtype=torch.float64)
    table = RegularGridCoefficient({"x": nodes}, nodes.square().add(1).mul(K0)).double()
    gauss, weights = np.polynomial.legendre.leggauss(6)  # type: ignore[no-untyped-call]
    h = 1 / cells
    # Six-node quadrature is exact for the degree-six squared cell residual.
    points = (np.arange(cells)[:, None] + (gauss + 1) / 2) * h
    quadrature = np.broadcast_to(weights * h / 2, points.shape).ravel()
    x = torch.tensor(points.reshape(-1, 1), dtype=torch.float64, requires_grad=True)
    time = torch.zeros_like(x, requires_grad=True)
    context = CoefficientBank({"conductivity": table}).context(Coordinates({"x": x}, time))
    q, first, second = x.square() * (3 - 2 * x), 6 * x * (1 - x), 6 - 12 * x
    kappa = x.square().add(1).mul(K0)
    exact = time.neg().exp() * q
    source = time.neg().exp() * (-q - kappa * second - x.mul(2 * K0) * first)
    equation = Equation({"u": exact}, context)
    sampled_kappa = context.coefficients["conductivity"]
    residual = (
        equation.dt("u")
        - equation.divergence({"x": sampled_kappa * equation.diff("u", "x")})
        - source
    )
    cell_left = torch.tensor(np.repeat(np.arange(cells) * h, len(gauss))[:, None])
    delta = (x - cell_left) * (cell_left + h - x) * K0
    delta_prime = (2 * cell_left + h - 2 * x) * K0
    closed_residual = -(delta * second + delta_prime * first)
    torch.testing.assert_close(residual, closed_residual, rtol=1e-10, atol=1e-11)
    actual_kappa_gradient = Equation({"k": sampled_kappa}, context).diff("k", "x")
    torch.testing.assert_close(actual_kappa_gradient, (2 * cell_left + h) * K0, rtol=0, atol=1e-12)
    r = residual.detach().numpy().ravel()
    rms = math.sqrt(float(np.dot(quadrature, r * r)) * TIME_INTEGRAL)
    exact_squared = K0**2 * TIME_INTEGRAL * (2 / 5 * h**2 + 2 / 5 * h**4 - 4 / 35 * h**6)
    exact_rms = math.sqrt(exact_squared)
    assert abs(rms - exact_rms) <= 1e-12 + 1e-9 * exact_rms
    signed_mean = float(np.dot(quadrature, r))
    assert abs(signed_mean) <= 1e-12
    mid = nodes[:-1].add(h / 2).reshape(-1, 1)
    middle = table(Coordinates({"x": mid}, torch.zeros_like(mid)))
    max_error = float((middle - mid.square().add(1).mul(K0)).abs().max())
    assert abs(max_error - K0 * h**2 / 4) <= 1e-15
    # Continuous error bound for the other PDE's exact weak solution;
    # this is not a measured neural-network solution error.
    solution_bound = max_error / math.sqrt(K0) * math.sqrt(6 / 5 * TIME_INTEGRAL)
    baseline = (
        equation.dt("u")
        - kappa * equation.diff("u", "x", 2)
        - x.mul(2 * K0) * equation.diff("u", "x")
        - source
    )
    baseline_max = float(baseline.detach().abs().max())
    assert baseline_max <= 1e-11
    return dict(
        cells=cells,
        h=h,
        coefficient_linf=max_error,
        coefficient_linf_exact=K0 * h**2 / 4,
        reference_residual_rms=rms,
        reference_residual_rms_exact=exact_rms,
        residual_formula_absolute_error=abs(rms - exact_rms),
        reference_residual_spatial_mean=signed_mean,
        formula_reference_residual_max=baseline_max,
        exact_solution_final_l2_upper_bound=solution_bound,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    torch.set_num_threads(1)
    rows = [measure(n) for n in (4, 8, 16, 32, 64, 128, 256)]
    for index, row in enumerate(rows):
        row["residual_observed_order"] = (
            math.log2(
                cast(float, rows[index - 1]["reference_residual_rms"])
                / cast(float, row["reference_residual_rms"])
            )
            if index
            else 0.0
        )
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"CPU coefficient refinements passed: {len(rows)}; no model training or solution-error measurement"
    )


if __name__ == "__main__":
    main()
