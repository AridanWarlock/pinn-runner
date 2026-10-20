"""Check the steep-front derivative via independent Gaussian-kernel moments."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

source = Path(__file__).with_name("2026-10-20-unforced-burgers-reference-audit.py")
spec = importlib.util.spec_from_file_location("burgers_reference", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def kernel_moments(x, t, nu):
    a = 1 / (2 * np.pi * nu)
    scale = np.sqrt(4 * nu * t)
    z = np.linspace(-12, 12, 2049)
    shift = np.max(-z * z - a * np.cos(np.pi * (x + scale * z)))

    def density(z):
        return np.exp(-z * z - a * np.cos(np.pi * (x + scale * z)) - shift)

    def integral(f):
        return sum(
            quad(lambda z: f(z) * density(z), k, k + 1, epsabs=1e-13, epsrel=1e-13)[0]
            for k in range(-12, 12)
        )

    normalizer = integral(lambda z: 1)
    mean = integral(lambda z: z) / normalizer
    variance = integral(lambda z: (z - mean) ** 2) / normalizer
    return -2 * np.sqrt(nu / t) * mean, (1 - 2 * variance) / t


rows = []
for nu in (0.1, 0.01):
    for x, t in (
        (0, 0.1),
        (0, 0.5),
        (0, 1),
        (0.01, 0.5),
        (0.01, 1),
        (-0.01, 1),
        (0.05, 0.5),
        (0.05, 1),
        (1, 0.5),
    ):
        u, ux = kernel_moments(x, t, nu)
        first = float(module.reference(x, t, nu))
        slope = float(module.reference(x, t, nu, derivative=True))
        h = 0.0001
        fd = float(
            (
                module.reference(x - 2 * h, t, nu)
                - 8 * module.reference(x - h, t, nu)
                + 8 * module.reference(x + h, t, nu)
                - module.reference(x + 2 * h, t, nu)
            )
            / (12 * h)
        )
        rows.append(
            dict(
                nu=nu,
                x=x,
                t=t,
                field=first,
                slope=slope,
                kernel_field=u,
                kernel_slope=ux,
                field_difference=abs(first - u),
                slope_difference=abs(slope - ux),
                finite_difference_error=abs(slope - fd),
            )
        )
base = Path(__file__).with_name("2026-10-20-unforced-burgers-gradient")
with base.with_suffix(".csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
summary = dict(
    points=len(rows),
    networks=0,
    optimizer_steps=0,
    field_difference_max=max(r["field_difference"] for r in rows),
    slope_difference_max=max(r["slope_difference"] for r in rows),
    finite_difference_error_max=max(r["finite_difference_error"] for r in rows),
    scope="Independent kernel moments at selected steep-front and boundary points; not a continuous certificate",
)
base.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
