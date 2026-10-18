"""CPU-only analytical preparation; no training or TASK-33 acceptance claim."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pinnstorch.api import Coordinates, Equation  # noqa: E402
from pinnstorch.coefficients import RegularGridCoefficient  # noqa: E402
from pinnstorch.runner.coefficients import CoefficientBank  # noqa: E402

K0 = 1 / 8


def closed(c: Coordinates, beta: int):
    xs = list(c.spatial.values())
    q = [x.square() * (3 - 2 * x) for x in xs]
    first = [6 * x * (1 - x) for x in xs]
    second = [6 - 12 * x for x in xs]
    product = math.prod(q)
    gradients = [
        first[i] * math.prod(q[j] for j in range(len(q)) if j != i) for i in range(len(q))
    ]
    diagonal = [
        second[i] * math.prod(q[j] for j in range(len(q)) if j != i) for i in range(len(q))
    ]
    conductivity = K0 * (1 + beta * sum(xs) / len(xs))
    wrong = K0 * beta / len(xs) * sum(gradients) * (-c.t).exp()
    source = (-c.t).exp() * (-product - conductivity * sum(diagonal)) - wrong
    divergence = conductivity * sum(diagonal) + K0 * beta / len(xs) * sum(gradients)
    return (
        (-c.t).exp() * product,
        source,
        conductivity,
        wrong,
        c.t * product,
        product - c.t * divergence,
    )


def samples(dimension: int):
    nodes, weights = np.polynomial.legendre.leggauss(8)
    nodes, weights = (nodes + 1) / 2, weights / 2
    points = np.stack(np.meshgrid(*([nodes] * (dimension + 1)), indexing="ij"), axis=-1).reshape(
        -1, dimension + 1
    )
    volume = np.prod(
        np.stack(np.meshgrid(*([weights] * (dimension + 1)), indexing="ij"), axis=-1), axis=-1
    ).ravel()
    boundary = np.array(
        list(itertools.product(*([[0, 0.125, 0.25, 0.5, 0.75, 1]] * dimension), [0, 0.5, 1]))
    )
    return np.concatenate([points, boundary]), np.concatenate([volume, np.zeros(len(boundary))])


def probe(dimension: int, dtype: torch.dtype, beta: int, kind: str):
    points, weights = samples(dimension)
    axes = ("x", "y")[:dimension]
    spatial = {
        axis: torch.tensor(points[:, i : i + 1], dtype=dtype, requires_grad=True)
        for i, axis in enumerate(axes)
    }
    time = torch.tensor(points[:, -1:], dtype=dtype, requires_grad=True)
    if kind == "table":
        vectors = {
            axis: torch.tensor(
                [0, 0.125, 0.5, 1] if axis == "x" else [0, 0.25, 0.75, 1], dtype=torch.float64
            )
            for axis in axes
        }
        grids = torch.meshgrid(*vectors.values(), indexing="ij")
        provider = RegularGridCoefficient(vectors, K0 * (1 + beta * sum(grids) / dimension)).to(
            dtype=dtype
        )
    else:

        def provider(c):
            return K0 * (1 + beta * sum(c.spatial.values()) / dimension)

    context = CoefficientBank({"conductivity": provider}).context(Coordinates(spatial, time))
    reference, forcing, coefficient, wrong_reference, correction, correction_operator = closed(
        context, beta
    )
    actual_coefficient = context.coefficients["conductivity"]
    tolerance = (
        dict(rtol=1e-10, atol=1e-11) if dtype == torch.float64 else dict(rtol=2e-4, atol=5e-6)
    )
    torch.testing.assert_close(actual_coefficient, coefficient, **tolerance)
    errors = []
    for axis in axes:
        derivative = Equation({"k": actual_coefficient}, context).diff("k", axis)
        expected = torch.full_like(time, K0 * beta / dimension)
        torch.testing.assert_close(derivative, expected, **tolerance)
        errors.append(float((derivative - expected).abs().max()))
    source_integral = float(np.dot(weights, forcing.detach().double().numpy().ravel()))
    source_expected = -(1 - math.exp(-1)) / 2**dimension
    assert abs(source_integral - source_expected) <= tolerance["atol"]
    result = dict(source_integral_absolute_error=abs(source_integral - source_expected))
    for amplitude in (1.0, 0.7):
        theta = torch.tensor(amplitude, dtype=dtype, requires_grad=True)
        field = reference + (theta - 1) * correction
        equation = Equation({"T": field}, context)
        gradient = equation.gradient("T")
        flux = {axis: actual_coefficient * value for axis, value in gradient.items()}
        residual = equation.dt("T") - equation.divergence(flux) - forcing
        expected_residual = (theta - 1) * correction_operator
        torch.testing.assert_close(residual, expected_residual, **tolerance)
        if amplitude == 1:
            result["reference_residual_max"] = float(residual.abs().max())
            normal_errors = [
                gradient[axis][(spatial[axis] == 0) | (spatial[axis] == 1)].abs().max()
                for axis in axes
            ]
            result["boundary_normal_max"] = max(map(float, normal_errors))
            assert result["boundary_normal_max"] <= tolerance["atol"]
            wrong_residual = (
                equation.dt("T")
                - actual_coefficient * sum(equation.diff("T", axis, 2) for axis in axes)
                - forcing
            )
            torch.testing.assert_close(wrong_residual, wrong_reference, **tolerance)
            observed = math.sqrt(
                float(np.dot(weights, wrong_residual.detach().double().numpy().ravel() ** 2))
            )
            spatial_integral = 6 / 5 if dimension == 1 else 487 / 350
            exact = K0 * beta / dimension * math.sqrt((1 - math.exp(-2)) / 2 * spatial_integral)
            assert abs(observed - exact) <= tolerance["atol"]
            result.update(wrong_operator_rmse=observed, wrong_operator_rmse_exact=exact)
        else:
            actual = torch.autograd.grad(residual.square().mean(), theta, retain_graph=True)[0]
            expected = 2 * (theta - 1) * correction_operator.square().mean()
            torch.testing.assert_close(actual, expected, **tolerance)
            result["hard_probe_residual_identity_max"] = float(
                (residual - expected_residual).abs().max()
            )
            result["hard_probe_gradient_absolute_error"] = float((actual - expected).abs())
            result["hard_probe_initial_max"] = float((field - reference)[time == 0].abs().max())
            assert result["hard_probe_initial_max"] == 0
            for axis in axes:
                torch.testing.assert_close(
                    gradient[axis][(spatial[axis] == 0) | (spatial[axis] == 1)],
                    torch.zeros_like(gradient[axis][(spatial[axis] == 0) | (spatial[axis] == 1)]),
                    **tolerance,
                )
    return dict(
        dimension=dimension,
        dtype=str(dtype),
        beta=beta,
        coefficient=kind,
        coefficient_max_error=float((actual_coefficient - coefficient).abs().max()),
        coefficient_gradient_max_error=max(errors),
        **result,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    rows = [
        probe(d, dtype, beta, kind)
        for d, dtype, beta, kind in itertools.product(
            (1, 2), (torch.float32, torch.float64), (0, 1), ("formula", "table")
        )
    ]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CPU reference checks passed: {len(rows)}; no training, torch={torch.__version__}")


if __name__ == "__main__":
    main()
