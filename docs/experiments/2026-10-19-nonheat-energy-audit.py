"""CPU quadrature of transport/Burgers error-energy identities; no networks or training."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PREFIX = Path(__file__).with_name("2026-10-19-nonheat-energy")
z, weights = np.polynomial.legendre.leggauss(64)
x, weights = (z + 1) / 2, weights / 2
rows = []
for amplitude in (0.7, -0.4):
    for instant in (0, 0.13, 0.5):
        e = amplitude * instant * (1 + x)
        residual = amplitude * (1 + x) + amplitude * instant
        energy_and_outflow = amplitude**2 * instant * 7 / 3 + 0.5 * (2 * amplitude * instant) ** 2
        inflow_and_work = 0.5 * (amplitude * instant) ** 2 + float(np.dot(weights, residual * e))
        assert math.isclose(energy_and_outflow, inflow_and_work, rel_tol=1e-13, abs_tol=1e-14)
        rows.append(
            dict(
                pde="transport",
                amplitude=amplitude,
                time=instant,
                left=energy_and_outflow,
                right=inflow_and_work,
                absolute_error=abs(energy_and_outflow - inflow_and_work),
            )
        )
        phi = np.sin(math.pi * x) - 0.3 * np.sin(2 * math.pi * x)
        dx_phi = math.pi * np.cos(math.pi * x) - 0.6 * math.pi * np.cos(2 * math.pi * x)
        dxx_phi = -(math.pi**2) * np.sin(math.pi * x) + 1.2 * math.pi**2 * np.sin(2 * math.pi * x)
        exact = np.exp(-instant) * np.sin(math.pi * x)
        exact_dx = math.pi * np.exp(-instant) * np.cos(math.pi * x)
        e, et = amplitude * instant * phi, amplitude * phi
        ex, exx = amplitude * instant * dx_phi, amplitude * instant * dxx_phi
        residual = et + exact * ex + e * exact_dx + e * ex - 0.1 * exx
        left = float(np.dot(weights, et * e + 0.1 * ex**2 + 0.5 * exact_dx * e**2))
        right = float(np.dot(weights, residual * e))
        assert math.isclose(left, right, rel_tol=1e-13, abs_tol=1e-14)
        rows.append(
            dict(
                pde="burgers",
                amplitude=amplitude,
                time=instant,
                left=left,
                right=right,
                absolute_error=abs(left - right),
            )
        )
with PREFIX.with_suffix(".csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
critical_time = math.log(5 / math.pi)


def exponent(instant):
    return math.pi / 2 * (1 - math.exp(-instant)) - 0.1 * math.pi**2 * instant


summary = dict(
    device="cpu",
    models=0,
    optimizer_steps=0,
    quadrature_nodes=64,
    checks=len(rows),
    max_absolute_error=max(r["absolute_error"] for r in rows),
    burgers_viscosity=0.1,
    burgers_horizon=0.5,
    critical_time=critical_time,
    burgers_max_homogeneous_growth_bound=math.exp(exponent(critical_time)),
    burgers_final_homogeneous_growth_bound=math.exp(exponent(0.5)),
    burgers_bound_without_diffusion=math.exp(math.pi / 2 * (1 - math.exp(-0.5))),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limitations="Continuous smooth error identities with stated boundary assumptions. No trained PINN accuracy or continuous certification from finite holdouts.",
)
PREFIX.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
