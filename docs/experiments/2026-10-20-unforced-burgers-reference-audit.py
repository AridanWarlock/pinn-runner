"""Positive Cole-Hopf quadrature for an unforced Burgers proposal, with no network."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import roots_hermite


@lru_cache
def rule(order):
    nodes, weights = roots_hermite(order)
    positive = weights > 0
    return nodes[positive], np.log(weights[positive])


def reference(x, t, nu, order=512, derivative=False):
    x, t = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(t, dtype=np.float64))
    z, logw = rule(order)
    y = x[..., None] + np.sqrt(4 * nu * t[..., None]) * z
    exponent = logw - np.cos(np.pi * y) / (2 * np.pi * nu)
    exponent -= exponent.max(axis=-1, keepdims=True)
    weights = np.exp(exponent)
    sine = np.sin(np.pi * y)
    total = weights.sum(axis=-1)
    mean_sine = (weights * sine).sum(axis=-1) / total
    if derivative:
        variance = (weights * (sine - mean_sine[..., None]) ** 2).sum(axis=-1) / total
        return -(weights * np.pi * np.cos(np.pi * y)).sum(axis=-1) / total - variance / (2 * nu)
    return -mean_sine


def adaptive(x, t, nu):
    a = 1 / (2 * np.pi * nu)
    sigma = np.sqrt(4 * nu * t)
    z = np.linspace(-12, 12, 2049)
    shift = float(np.max(-z * z - a * np.cos(np.pi * (x + sigma * z))))

    def density(z):
        y = x + sigma * z
        return np.exp(-z * z - a * np.cos(np.pi * y) - shift)

    denominator = sum(
        quad(density, k, k + 1, epsabs=1e-13, epsrel=1e-13)[0] for k in range(-12, 12)
    )
    numerator = sum(
        quad(
            lambda z: -np.sin(np.pi * (x + sigma * z)) * density(z),
            k,
            k + 1,
            epsabs=1e-13,
            epsrel=1e-13,
        )[0]
        for k in range(-12, 12)
    )
    return numerator / denominator


def main():
    rows = []
    for nu in (0.1, 0.01):
        x = np.linspace(-1, 1, 201)[:, None]
        t = np.linspace(0, 1, 51)[None, :]
        controls = reference(x, t, nu, 512)
        for order in (32, 64, 128, 256):
            values = reference(x, t, nu, order)
            rows.append(
                dict(
                    kind="order",
                    nu=nu,
                    order=order,
                    max_error=float(abs(values - controls).max()),
                    rms_error=float(np.sqrt(np.mean((values - controls) ** 2))),
                )
            )
        for xx, tt in (
            (0, 0),
            (0.05, 0.1),
            (0.05, 0.5),
            (0.05, 1),
            (0.3, 0.3),
            (0.7, 1),
            (-0.05, 1),
            (1, 0.5),
        ):
            value = float(reference(xx, tt, nu))
            independent = adaptive(xx, tt, nu)
            rows.append(
                dict(
                    kind="adaptive",
                    nu=nu,
                    x=xx,
                    t=tt,
                    value=value,
                    independent=independent,
                    max_error=abs(value - independent),
                )
            )
        for step in (0.002, 0.001, 0.0005):
            defects = []
            for xx, tt in ((0.05, 0.1), (0.05, 0.5), (0.05, 1), (0.3, 0.3), (0.7, 1)):

                def f(xx, tt):
                    return float(reference(xx, tt, nu))

                u = f(xx, tt)
                ux = (
                    f(xx - 2 * step, tt)
                    - 8 * f(xx - step, tt)
                    + 8 * f(xx + step, tt)
                    - f(xx + 2 * step, tt)
                ) / (12 * step)
                uxx = (
                    -f(xx + 2 * step, tt)
                    + 16 * f(xx + step, tt)
                    - 30 * u
                    + 16 * f(xx - step, tt)
                    - f(xx - 2 * step, tt)
                ) / (12 * step * step)
                ut = (
                    f(xx, tt - 2 * step)
                    - 8 * f(xx, tt - step)
                    + 8 * f(xx, tt + step)
                    - f(xx, tt + 2 * step)
                ) / (12 * step)
                defects.append(abs(ut + u * ux - nu * uxx))
            rows.append(
                dict(kind="pde_finite_difference", nu=nu, step=step, max_error=max(defects))
            )
        energy = 0.5 * np.trapz(controls**2, x[:, 0], axis=0)
        mass = np.trapz(controls, x[:, 0], axis=0)
        rows.append(
            dict(
                kind="invariants",
                nu=nu,
                energy_initial=float(energy[0]),
                energy_final=float(energy[-1]),
                energy_max_increase=float(max(0, np.max(np.diff(energy)))),
                mass_max=float(abs(mass).max()),
                center_slope_final=float(reference(0, 1, nu, derivative=True)),
            )
        )
        derivative_errors = []
        h = 0.0001
        for xx, tt in ((0, 0), (0.05, 0.1), (0.05, 0.5), (0.05, 1), (0.3, 0.3), (0.7, 1)):
            estimate = (
                reference(xx - 2 * h, tt, nu)
                - 8 * reference(xx - h, tt, nu)
                + 8 * reference(xx + h, tt, nu)
                - reference(xx + 2 * h, tt, nu)
            ) / (12 * h)
            derivative_errors.append(abs(float(estimate - reference(xx, tt, nu, derivative=True))))
        rows.append(dict(kind="slope_control", nu=nu, step=h, max_error=max(derivative_errors)))
        rows.append(
            dict(
                kind="data",
                nu=nu,
                max_error=float(
                    max(
                        abs(controls[:, 0] + np.sin(np.pi * x[:, 0])).max(),
                        abs(controls[0]).max(),
                        abs(controls[-1]).max(),
                    )
                ),
                amplitude_max=float(abs(controls).max()),
            )
        )
    base = Path(__file__).with_name("2026-10-20-unforced-burgers-reference")
    with base.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        nu=[0.1, 0.01],
        domain=[-1, 1],
        time=[0, 1],
        source=0,
        networks=0,
        optimizer_steps=0,
        adaptive_checks=16,
        slope_finite_difference_max=max(
            r["max_error"] for r in rows if r["kind"] == "slope_control"
        ),
        invariants=[r for r in rows if r["kind"] == "invariants"],
        adaptive_max_error=max(r["max_error"] for r in rows if r["kind"] == "adaptive"),
        order_256_to_512_max=max(
            r["max_error"] for r in rows if r["kind"] == "order" and r["order"] == 256
        ),
        pde_smallest_step_max=max(
            r["max_error"]
            for r in rows
            if r["kind"] == "pde_finite_difference" and r["step"] == 0.0005
        ),
        data_max_error=max(r["max_error"] for r in rows if r["kind"] == "data"),
    )
    base.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
