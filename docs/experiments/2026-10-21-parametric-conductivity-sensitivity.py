"""Exact parameter sensitivities validate the independent diffusion reference."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "docs/experiments/2026-10-21-parametric-conductivity"
spec = importlib.util.spec_from_file_location("reference", f"{PREFIX}-reference.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ALPHA = np.pi**2 / 8


def exact(x, t, axis):
    if axis == 0:
        return -(np.exp(-ALPHA * t) - np.exp(-4 * ALPHA * t)) * np.sin(2 * np.pi * x) / 3
    return (
        -ALPHA * t * np.exp(-ALPHA * t) * np.sin(np.pi * x) / 2
        - 3 * (np.exp(-ALPHA * t) - np.exp(-9 * ALPHA * t)) * np.sin(3 * np.pi * x) / 16
    )


def main():
    points = np.random.default_rng(20261021).uniform(0.05, 0.95, size=(37, 2))
    rows = []
    for axis in (0, 1):
        x = torch.tensor(points[:, 0:1], dtype=torch.float64, requires_grad=True)
        t = torch.tensor(points[:, 1:2], dtype=torch.float64, requires_grad=True)
        alpha = torch.pi**2 / 8
        if axis == 0:
            value = (
                -(torch.exp(-alpha * t) - torch.exp(-4 * alpha * t))
                * torch.sin(2 * torch.pi * x)
                / 3
            )
        else:
            value = (
                -alpha * t * torch.exp(-alpha * t) * torch.sin(torch.pi * x) / 2
                - 3
                * (torch.exp(-alpha * t) - torch.exp(-9 * alpha * t))
                * torch.sin(3 * torch.pi * x)
                / 16
            )
        sx, st = torch.autograd.grad(value, (x, t), torch.ones_like(value), create_graph=True)
        sxx = torch.autograd.grad(sx, x, torch.ones_like(sx), create_graph=True)[0]
        j = axis + 1
        source = (
            -(torch.pi**2 / 16)
            * torch.exp(-alpha * t)
            * (
                (j - 1) * torch.sin((j - 1) * torch.pi * x)
                + (j + 1) * torch.sin((j + 1) * torch.pi * x)
            )
        )
        residual = float((st - sxx / 8 - source).abs().max().detach())
        for n in (512, 1024):
            grid = np.linspace(0, 1, n + 1)[:, None]
            analytic = exact(grid, module.TIMES, axis)
            for h in (1e-3, 5e-4):
                pair = [0.0, 0.0]
                pair[axis] = h
                plus = module.reference(n, *pair)[1]
                pair[axis] = -h
                minus = module.reference(n, *pair)[1]
                difference = (plus - minus) / (2 * h)
                error = float(np.abs(difference - analytic).max())
                rows.append(
                    dict(
                        parameter="a" if axis == 0 else "b",
                        intervals=n,
                        parameter_step=h,
                        maximum_analytic_error=error,
                        analytic_sensitivity_pde_max=residual,
                        initial_boundary_error=float(
                            max(np.abs(analytic[:, 0]).max(), np.abs(analytic[[0, -1]]).max())
                        ),
                        passed=bool(residual <= 1e-12 and (n != 1024 or error <= 2e-6)),
                    )
                )

    def integral(rate):
        return -np.expm1(-rate) / rate

    rate = 2 * ALPHA
    time_squared = (2 - np.exp(-rate) * (rate**2 + 2 * rate + 2)) / rate**3
    norm_a_squared = (integral(2 * ALPHA) - 2 * integral(5 * ALPHA) + integral(8 * ALPHA)) / 18
    norm_b_squared = (
        ALPHA**2 * time_squared / 8
        + 9 * (integral(2 * ALPHA) - 2 * integral(10 * ALPHA) + integral(18 * ALPHA)) / 512
    )
    independent_a = quad(
        lambda t: (np.exp(-ALPHA * t) - np.exp(-4 * ALPHA * t)) ** 2 / 18,
        0,
        1,
        epsabs=1e-14,
        epsrel=1e-13,
    )[0]
    independent_b = quad(
        lambda t: (
            ALPHA**2 * t**2 * np.exp(-2 * ALPHA * t) / 8
            + 9 * (np.exp(-ALPHA * t) - np.exp(-9 * ALPHA * t)) ** 2 / 512
        ),
        0,
        1,
        epsabs=1e-14,
        epsrel=1e-13,
    )[0]
    norm_error = max(abs(norm_a_squared - independent_a), abs(norm_b_squared - independent_b))
    with Path(f"{PREFIX}-sensitivity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        protocol_sha256=hashlib.sha256(
            (ROOT / "docs/parametric-conductivity-sensitivity-protocol-2026-10-21.md").read_bytes()
        ).hexdigest(),
        optimizer_steps=0,
        rows=len(rows),
        norm_a=float(np.sqrt(norm_a_squared)),
        norm_b=float(np.sqrt(norm_b_squared)),
        norm_independent_difference=float(norm_error),
        all_passed=bool(all(r["passed"] for r in rows) and norm_error <= 1e-12),
    )
    Path(f"{PREFIX}-sensitivity.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2))
    print(rows)
    assert result["all_passed"], "Preserve failing sensitivities"


if __name__ == "__main__":
    main()
