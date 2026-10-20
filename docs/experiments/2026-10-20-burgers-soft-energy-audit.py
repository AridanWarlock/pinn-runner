"""Independent FP64 quadrature of the nonhomogeneous-boundary Burgers error identity."""

import csv
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

nodes, weights = leggauss(64)
x, weights = (nodes + 1) / 2, weights / 2
nu = 0.1
rows = []
for a, c in ((0.4, 0.7), (-0.3, 0.2), (0.01, 0)):
    for t in (0.05, 0.13, 0.5):
        e = a * t * (1 + x) + c * t * np.sin(np.pi * x)
        et = a * (1 + x) + c * np.sin(np.pi * x)
        ex = a * t + c * t * np.pi * np.cos(np.pi * x)
        exx = -c * t * np.pi**2 * np.sin(np.pi * x)
        u = np.exp(-t) * np.sin(np.pi * x)
        ux = np.exp(-t) * np.pi * np.cos(np.pi * x)
        residual = et + u * ex + e * ux + e * ex - nu * exx
        lhs = float(weights @ (e * et + nu * ex**2 + 0.5 * ux * e**2))
        source_work = float(weights @ (residual * e))
        wall = np.array([0.0, 1.0])
        ew = a * t * (1 + wall) + c * t * np.sin(np.pi * wall)
        exw = a * t + c * t * np.pi * np.cos(np.pi * wall)
        uw = np.exp(-t) * np.sin(np.pi * wall)
        flux = 0.5 * uw * ew**2 + ew**3 / 3 - nu * ew * exw
        boundary_jump = float(flux[1] - flux[0])
        defect = abs(lhs - source_work + boundary_jump)
        assert defect < 2e-14
        rows.append(
            dict(
                a=a,
                c=c,
                t=t,
                lhs=lhs,
                residual_work=source_work,
                boundary_jump=boundary_jump,
                identity_defect=defect,
                wrong_homogeneous_defect=abs(lhs - source_work),
            )
        )
base = Path(__file__).with_name("2026-10-20-burgers-soft-energy")
with base.with_suffix(".csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = dict(
    checks=len(rows),
    dtype="float64",
    quadrature_order=64,
    optimizer_steps=0,
    networks=0,
    max_identity_defect=max(r["identity_defect"] for r in rows),
    max_wrong_homogeneous_defect=max(r["wrong_homogeneous_defect"] for r in rows),
    nonzero_boundary_checks=sum(abs(r["boundary_jump"]) > 1e-10 for r in rows),
)
base.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
