"""CPU boundary-jet diagnostic; the cosine probe is not a trained network."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.audit_variable_conductivity import make_factory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for beta, kind in itertools.product((0, 1), ("formula", "table")):
        factory = make_factory(args.output / f"beta{beta}-{kind}", 1, beta, kind, 19)
        factory.adapter.coefficients.double()
        x = torch.tensor([[0.0], [1.0]], dtype=torch.float64, requires_grad=True)
        time = torch.full_like(x, 0.37, requires_grad=True)
        context = factory.adapter.context([x], time)
        q = factory.problem.initial(context)["u"]
        exact = time.neg().exp() * q
        r_exact = factory.adapter.residuals({"u": exact}, x, time)["heat"]
        r_exact_x = torch.autograd.grad(r_exact.sum(), x, create_graph=True)[0]
        assert float(r_exact.detach().abs().max()) <= 1e-11
        assert float(r_exact_x.detach().abs().max()) <= 1e-11
        for value in (0.0, 0.7):
            amplitude = torch.tensor(value, dtype=torch.float64, requires_grad=True)
            probe = q + amplitude * time * (math.pi * x).cos()
            r = factory.adapter.residuals({"u": probe}, x, time)["heat"]
            rx = torch.autograd.grad(r.sum(), x, create_graph=True)[0]
            slope = torch.stack(
                [torch.autograd.grad(rx[i].sum(), amplitude, retain_graph=True)[0] for i in (0, 1)]
            ).reshape(-1, 1)
            kappa = (1 + beta * x) / 8
            expected = (time.neg().exp() - 1) * (-12 * kappa + 2 * beta / 8 * (6 - 12 * x))
            expected = (
                expected + 2 * beta / 8 * amplitude * time * math.pi**2 * (math.pi * x).cos()
            )
            expected_slope = 2 * beta / 8 * time * math.pi**2 * (math.pi * x).cos()
            torch.testing.assert_close(rx, expected, rtol=1e-10, atol=1e-11)
            torch.testing.assert_close(slope, expected_slope, rtol=1e-10, atol=1e-11)
            derivative = torch.autograd.grad(probe.sum(), x, create_graph=True)[0]
            flux_max = float((kappa * derivative).detach().abs().max())
            assert flux_max <= 1e-11
            rows.append(
                dict(
                    beta=beta,
                    representation=kind,
                    amplitude=value,
                    time=0.37,
                    exact_residual_max=float(r_exact.detach().abs().max()),
                    exact_residual_x_max=float(r_exact_x.detach().abs().max()),
                    probe_residual_x_lower=float(rx[0].detach()),
                    probe_residual_x_upper=float(rx[1].detach()),
                    amplitude_slope_lower=float(slope[0]),
                    amplitude_slope_upper=float(slope[1]),
                    residual_x_identity_max=float((rx - expected).detach().abs().max()),
                    amplitude_slope_identity_max=float(
                        (slope - expected_slope).detach().abs().max()
                    ),
                    physical_flux_max=flux_max,
                )
            )
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("CPU boundary identities passed:8; no training or error-lower-bound claim")


if __name__ == "__main__":
    main()
