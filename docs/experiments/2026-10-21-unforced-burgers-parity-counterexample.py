"""A smooth same-data counterexample to universal residual improvement by parity."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss


def fields(x, t, c=1.0, p=12):
    z = -np.sin(np.pi * x) + c * t * x * (1 - x * x)
    zx = -np.pi * np.cos(np.pi * x) + c * t * (1 - 3 * x * x)
    zxx = np.pi**2 * np.sin(np.pi * x) - 6 * c * t * x
    rz = c * x * (1 - x * x) + z * zx - 0.01 * zxx
    b = (1 - x * x) ** p
    bx = -2 * p * x * (1 - x * x) ** (p - 1)
    bxx = -2 * p * (1 - x * x) ** (p - 1) + 4 * p * (p - 1) * x * x * (1 - x * x) ** (p - 2)
    a = t * b
    ax = t * bx
    linear = b + z * ax + zx * a - 0.01 * t * bxx
    return z, zx, zxx, rz, a, ax, linear


def main():
    # Reproduce the finite exploratory matrix; none of these rows is a training run.
    x = np.linspace(-1, 1, 801)[:, None]
    t = np.linspace(0, 1, 401)[None, :]

    def trapezoid(v):
        return float(np.trapz(np.trapz(v, x[:, 0], axis=0), t[0]))

    search = []
    for c in (1, 5, 10, 20, 40):
        for p in (3, 4, 6, 8, 12):
            _, _, _, rz, a, ax, linear = fields(x, t, c, p)
            q = trapezoid(linear**2 + 2 * rz * a * ax)
            fourth = trapezoid((a * ax) ** 2)
            search.append(dict(c=c, p=p, q=q, fourth=fourth, q_negative=q < 0))
    epsilon = 0.4
    rows = []
    for order in (64, 128):
        nodes, weights = leggauss(order)
        xg, tg = nodes[:, None], (nodes[None, :] + 1) / 2
        weight = weights[:, None] * weights[None, :] / 2
        z, _, _, rz, a, ax, linear = fields(xg, tg)
        raw = rz + epsilon * linear + epsilon**2 * a * ax

        def integrate(v):
            return float(np.sum(weight * v))

        q = integrate(linear**2 + 2 * rz * a * ax)
        fourth = integrate((a * ax) ** 2)
        iz, iv = integrate(rz**2), integrate(raw**2)
        predicted = epsilon**2 * q + epsilon**4 * fourth
        rows.append(
            dict(
                order=order,
                c=1,
                p=12,
                epsilon=epsilon,
                q=q,
                fourth=fourth,
                raw_integral_squared_residual=iv,
                projected_integral_squared_residual=iz,
                raw_pde_rms=(iv / 2) ** 0.5,
                projected_pde_rms=(iz / 2) ** 0.5,
                projection_rms_ratio=(iz / iv) ** 0.5,
                raw_minus_projected=iv - iz,
                identity_absolute_defect=abs((iv - iz) - predicted),
                projection_worsens=iz > iv,
            )
        )
    points = np.random.default_rng(20261022).uniform(size=(17, 2))
    points[:, 0] = 1.6 * points[:, 0] - 0.8
    points[:, 1] = 0.1 + 0.8 * points[:, 1]
    xc = points[:, 0:1]
    tc = points[:, 1:2]

    def value(x, t):
        return -np.sin(np.pi * x) + t * x * (1 - x * x) + epsilon * t * (1 - x * x) ** 12

    _, _, _, rz, a, ax, linear = fields(xc, tc)
    analytic = rz + epsilon * linear + epsilon**2 * a * ax
    xt, tt = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (xc, tc)]
    v = -torch.sin(torch.pi * xt) + tt * xt * (1 - xt * xt) + epsilon * tt * (1 - xt * xt) ** 12
    vx, vt = torch.autograd.grad(v, (xt, tt), torch.ones_like(v), create_graph=True)
    vxx = torch.autograd.grad(vx, xt, torch.ones_like(vx))[0]
    ad = (vt + v * vx - 0.01 * vxx).detach().numpy()
    ad_error = float(np.max(np.abs(ad - analytic)))
    fd = []
    for h in (1e-3, 5e-4):
        dx = (
            value(xc - 2 * h, tc)
            - 8 * value(xc - h, tc)
            + 8 * value(xc + h, tc)
            - value(xc + 2 * h, tc)
        ) / (12 * h)
        dt = (
            value(xc, tc - 2 * h)
            - 8 * value(xc, tc - h)
            + 8 * value(xc, tc + h)
            - value(xc, tc + 2 * h)
        ) / (12 * h)
        dxx = (
            -value(xc + 2 * h, tc)
            + 16 * value(xc + h, tc)
            - 30 * value(xc, tc)
            + 16 * value(xc - h, tc)
            - value(xc - 2 * h, tc)
        ) / (12 * h * h)
        error = float(np.max(np.abs(dt + value(xc, tc) * dx - 0.01 * dxx - analytic)))
        fd.append(dict(h=h, maximum_absolute_error=error, passed=error <= 1e-7))
    base = Path(__file__).with_name("2026-10-21-unforced-burgers-parity-counterexample")
    for suffix, data in (("-search.csv", search), (".csv", rows)):
        target = Path(str(base) + suffix)
        assert not target.exists()
        with target.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    summary = dict(
        c=1,
        p=12,
        epsilon=epsilon,
        viscosity=0.01,
        optimizer_steps=0,
        exploratory_candidates=25,
        selected_after_exploration=True,
        quadrature_orders=[64, 128],
        quadrature_absolute_rms_difference=max(
            abs(rows[0][k] - rows[1][k]) for k in ("raw_pde_rms", "projected_pde_rms")
        ),
        counterexample_confirmed=all(r["projection_worsens"] for r in rows),
        maximum_identity_defect=max(r["identity_absolute_defect"] for r in rows),
        autograd_maximum_absolute_error=ad_error,
        autograd_passed=ad_error <= 1e-12,
        finite_difference_controls=fd,
        scope="Constructive same-IC/BC smooth example; no PINN training, not a counterexample to all registered quality goals or an interval certificate.",
    )
    assert not base.with_suffix(".json").exists()
    base.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
