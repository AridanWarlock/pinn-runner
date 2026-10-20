"""Audit continuous error identities on explicit perturbations; no PINN or training."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import eigh_tridiagonal
from scipy.special import roots_legendre

SOURCE = Path(__file__).with_name("2026-10-20-unforced-burgers-reference-audit.py")
SPEC = importlib.util.spec_from_file_location("burgers_reference", SOURCE)
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)
AUDIT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
REFERENCE_SHA = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
BASE = Path(__file__).with_name("2026-10-20-unforced-burgers-stability")


def values(x, t, nu, derivative=False):
    """Bound peak memory while retaining the independent FP64 reference."""
    xx, tt = np.broadcast_arrays(np.asarray(x), np.asarray(t))
    result = np.empty(xx.size)
    for start in range(0, xx.size, 256):
        stop = start + 256
        result[start:stop] = REFERENCE.reference(
            xx.flat[start:stop], tt.flat[start:stop], nu, derivative=derivative
        )
    return result.reshape(xx.shape)


def coefficients(nu, nx, nt):
    x = np.linspace(-1, 1, nx)[:, None]
    t = np.linspace(0, 1, nt)
    ux = values(x, t[None, :], nu, derivative=True)
    # This sampled maximum is a diagnostic, NOT a rigorous continuous supremum.
    negative_slope = np.maximum(0, -ux.min(axis=0))
    rate = negative_slope / 2 - nu * np.pi**2 / 4
    integral = np.r_[0.0, cumulative_trapezoid(rate, t)]
    # Bound for a space-time L2 residual after Cauchy-Schwarz in time.
    kernel = np.exp(integral[-1] - integral)
    spacing = float(x[1, 0] - x[0, 0])
    off_diagonal = np.full(nx - 3, nu / spacing**2)
    sharp_rate = np.array(
        [
            eigh_tridiagonal(
                -2 * nu / spacing**2 - ux[1:-1, j] / 2,
                off_diagonal,
                select="i",
                select_range=(nx - 3, nx - 3),
                eigvals_only=True,
            )[0]
            for j in range(nt)
        ]
    )
    discrete_poincare = 4 * nu / spacing**2 * np.sin(np.pi / (2 * (nx - 1))) ** 2
    assert np.max(sharp_rate - (negative_slope / 2 - discrete_poincare)) < 1e-8
    sharp_integral = np.r_[0.0, cumulative_trapezoid(sharp_rate, t)]
    sharp_kernel = np.exp(sharp_integral[-1] - sharp_integral)
    peak_x, peak_t = np.unravel_index(np.argmin(ux), ux.shape)
    uniform_rate = np.pi / 2 + 1 / (4 * nu) - nu * np.pi**2 / 4
    return dict(
        kind="coefficients",
        nu=nu,
        nx=nx,
        nt=nt,
        negative_slope_max=float(negative_slope.max()),
        steepest_x=float(x[peak_x, 0]),
        steepest_time=float(t[peak_t]),
        growth_integral_final=float(integral[-1]),
        initial_error_multiplier=float(np.exp(integral[-1])),
        residual_l2_multiplier=float(np.sqrt(np.trapz(kernel**2, t))),
        rayleigh_rate_max=float(sharp_rate.max()),
        rayleigh_integral_final=float(sharp_integral[-1]),
        rayleigh_initial_multiplier=float(np.exp(sharp_integral[-1])),
        rayleigh_residual_l2_multiplier=float(np.sqrt(np.trapz(sharp_kernel**2, t))),
        analytic_uniform_rate=float(uniform_rate),
        analytic_initial_multiplier=float(np.exp(uniform_rate)),
        analytic_residual_l2_multiplier=float(
            np.sqrt(np.expm1(2 * uniform_rate) / (2 * uniform_rate))
        ),
    )


def quadrature(order):
    # Split at the interior zero of e so |e| is smooth on each integration panel.
    nodes, weights = roots_legendre(order)
    return np.r_[(nodes - 1) / 2, (nodes + 1) / 2], np.tile(weights / 2, 2)


def perturbation(x, t, amplitude, harmonic):
    shape = np.sin(np.pi * x) + harmonic * np.sin(2 * np.pi * x)
    first = np.pi * (np.cos(np.pi * x) + 2 * harmonic * np.cos(2 * np.pi * x))
    second = -(np.pi**2) * (np.sin(np.pi * x) + 4 * harmonic * np.sin(2 * np.pi * x))
    return amplitude * t * shape, amplitude * shape, amplitude * t * first, amplitude * t * second


def balances(nu, amplitude, harmonic, order, nt):
    x, weights = quadrature(order)
    t = np.linspace(0, 1, nt)
    xx, tt = x[:, None], t[None, :]
    u = values(xx, tt, nu)
    ux = values(xx, tt, nu, derivative=True)
    e, et, ex, exx = perturbation(xx, tt, amplitude, harmonic)
    residual = et + u * ex + ux * e + e * ex - nu * exx

    def integrate(a):
        return weights @ a

    energy_left = integrate(e * et) + nu * integrate(ex**2) + 0.5 * integrate(ux * e**2)
    energy_right = integrate(residual * e)
    # For |harmonic| < 1/2, sign(e) has only the one interior change at x=0.
    assert abs(harmonic) < 0.5
    sign = np.sign(e)
    # e=0 at t=0: its right-time sign is determined by e_t, not by the target norm.
    sign[:, 0] = np.sign(et[:, 0])
    signed_residual = integrate(sign * residual)
    l1_derivative = 4 * abs(amplitude) / np.pi
    boundary_loss = 2 * nu * np.pi * abs(amplitude) * t * (1 - 2 * harmonic)
    interior_loss = 2 * nu * np.pi * abs(amplitude) * t * (1 + 2 * harmonic)
    l1_identity = signed_residual - boundary_loss - interior_loss
    r_l1 = integrate(abs(residual))
    r_l2_squared = integrate(residual**2)
    accumulated_l1 = np.r_[0.0, cumulative_trapezoid(r_l1, t)]
    actual_l1 = 4 * abs(amplitude) * t / np.pi
    actual_l2 = abs(amplitude) * t * np.sqrt(1 + harmonic**2)
    field_amplitude = 1 + abs(amplitude) * (1 + abs(harmonic))
    # |u| <= 1 and |v| <= field_amplitude, both continuous analytic bounds here.
    error_amplitude_bound = 1 + field_amplitude
    l1_to_l2 = np.sqrt(error_amplitude_bound * accumulated_l1[-1])
    u_final_l2 = np.sqrt(integrate(u**2)[-1])
    # Conditional sufficient residual RMS for a relative final L2 target of .003.
    # Space-time volume is 2, and ||r||_L1 <= 2*RMS on [-1,1] x [0,1].
    sufficient_rms = (0.003 * u_final_l2) ** 2 / (2 * error_amplitude_bound)
    assert np.max(abs(energy_left - energy_right)) < 1e-11
    assert np.max(abs(l1_identity - l1_derivative)) < 1e-10
    assert np.min(accumulated_l1 - actual_l1) > -1e-10
    assert actual_l2[-1] <= l1_to_l2
    return dict(
        kind="perturbation",
        nu=nu,
        amplitude=amplitude,
        harmonic=harmonic,
        quadrature_per_half=order,
        nt=nt,
        energy_identity_error_max=float(abs(energy_left - energy_right).max()),
        l1_identity_error_max=float(abs(l1_identity - l1_derivative).max()),
        boundary_dissipation_final=float(boundary_loss[-1]),
        interior_dissipation_final=float(interior_loss[-1]),
        actual_l1_final=float(actual_l1[-1]),
        accumulated_residual_l1=float(accumulated_l1[-1]),
        actual_l2_final=float(actual_l2[-1]),
        l1_to_l2_bound=float(l1_to_l2),
        residual_rms=float(np.sqrt(np.trapz(r_l2_squared, t) / 2)),
        reference_l2_final=float(u_final_l2),
        field_amplitude_bound=float(field_amplitude),
        conditional_sufficient_rms=float(sufficient_rms),
    )


def finite_difference(nu, amplitude, harmonic, step):
    errors = []
    for x, t in ((0, 0.5), (0.01, 1), (-0.05, 0.5), (0.3, 0.3), (0.7, 1)):

        def field(x, t):
            return (
                float(REFERENCE.reference(x, t, nu)) + perturbation(x, t, amplitude, harmonic)[0]
            )

        v = field(x, t)
        vx = (
            field(x - 2 * step, t)
            - 8 * field(x - step, t)
            + 8 * field(x + step, t)
            - field(x + 2 * step, t)
        ) / (12 * step)
        vxx = (
            -field(x + 2 * step, t)
            + 16 * field(x + step, t)
            - 30 * v
            + 16 * field(x - step, t)
            - field(x - 2 * step, t)
        ) / (12 * step**2)
        vt = (
            field(x, t - 2 * step)
            - 8 * field(x, t - step)
            + 8 * field(x, t + step)
            - field(x, t + 2 * step)
        ) / (12 * step)
        u = float(REFERENCE.reference(x, t, nu))
        ux = float(REFERENCE.reference(x, t, nu, derivative=True))
        e, et, ex, exx = perturbation(x, t, amplitude, harmonic)
        explicit = et + u * ex + ux * e + e * ex - nu * exx
        errors.append(abs(vt + v * vx - nu * vxx - explicit))
    return dict(
        kind="finite_difference",
        nu=nu,
        amplitude=amplitude,
        harmonic=harmonic,
        step=step,
        points=len(errors),
        residual_error_max=max(errors),
    )


def peak_control(row):
    x, t, nu = row["steepest_x"], row["steepest_time"], row["nu"]
    exact = float(REFERENCE.reference(x, t, nu, derivative=True))
    controls = []
    for step in (0.0002, 0.0001, 0.00005):
        derivative = (
            REFERENCE.adaptive(x - 2 * step, t, nu)
            - 8 * REFERENCE.adaptive(x - step, t, nu)
            + 8 * REFERENCE.adaptive(x + step, t, nu)
            - REFERENCE.adaptive(x + 2 * step, t, nu)
        ) / (12 * step)
        controls.append(dict(step=step, slope=float(derivative), error=abs(derivative - exact)))
    assert controls[-1]["error"] < 1e-6
    return dict(nu=nu, x=x, t=t, slope=exact, independent_adaptive_finite_differences=controls)


def main():
    rows = []
    for nu in (0.1, 0.01):
        for nx, nt in ((401, 101), (801, 201), (1601, 401)):
            rows.append(coefficients(nu, nx, nt))
        for amplitude, harmonic in ((0.01, 0.0), (-0.05, -0.3)):
            for order, nt in ((128, 201), (256, 401)):
                rows.append(balances(nu, amplitude, harmonic, order, nt))
            for step in (0.002, 0.001, 0.0005):
                rows.append(finite_difference(nu, amplitude, harmonic, step))
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == AUDIT_SHA
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == REFERENCE_SHA
    for row in rows:
        if row.get("step") == 0.0005:
            assert row["residual_error_max"] < 1e-6
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with BASE.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        networks=0,
        optimizer_steps=0,
        rows=len(rows),
        coefficient_grids=6,
        balance_checks=8,
        finite_difference_checks=12,
        reference_sha256=REFERENCE_SHA,
        audit_sha256=AUDIT_SHA,
        energy_identity_error_max=max(r.get("energy_identity_error_max", 0) for r in rows),
        l1_identity_error_max=max(r.get("l1_identity_error_max", 0) for r in rows),
        finite_difference_error_max=max(r.get("residual_error_max", 0) for r in rows),
        coefficients=[r for r in rows if r["kind"] == "coefficients"],
        peak_controls=[peak_control(r) for r in rows if r.get("nx") == 1601],
        finest_balances=[r for r in rows if r.get("quadrature_per_half") == 256],
        scope="Continuous identities proved in the report; sampled slope maxima and numerical integrals are diagnostics, not certificates. No PINN accuracy or speed claim.",
    )
    BASE.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
