"""Resolve a smooth local error that small field norms can hide; no training."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre

SOURCE = Path(__file__).with_name("2026-10-20-unforced-burgers-reference-audit.py")
SPEC = importlib.util.spec_from_file_location("burgers_reference", SOURCE)
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)
AUDIT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
REFERENCE_SHA = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
BASE = Path(__file__).with_name("2026-10-20-unforced-burgers-layer")


def shape(x, delta):
    z = x / delta
    gaussian = np.exp(-(z**2) / 2)
    g = z * gaussian
    gx = (1 - z**2) * gaussian / delta
    gxx = (z**3 - 3 * z) * gaussian / delta**2
    return g * (1 - x**2), gx * (1 - x**2) - 2 * x * g, gxx * (1 - x**2) - 4 * x * gx - 2 * g


def reference(x, nu, derivative=False):
    result = np.empty_like(x)
    for start in range(0, len(x), 256):
        result[start : start + 256] = REFERENCE.reference(
            x[start : start + 256], 1.0, nu, derivative=derivative
        )
    return result


def calculate(nu, delta, order):
    nodes, weights = roots_legendre(order)
    # Explicit panels resolve the central bump even when delta=.0025.
    breaks = (-1, -0.1, 0, 0.1, 1)
    x = np.concatenate([(a + b) / 2 + (b - a) * nodes / 2 for a, b in zip(breaks, breaks[1:])])
    w = np.concatenate([(b - a) * weights / 2 for a, b in zip(breaks, breaks[1:])])
    u, ux = reference(x, nu), reference(x, nu, True)
    amplitude = 0.01
    psi, psix, psixx = shape(x, delta)
    e, ex = amplitude * psi, amplitude * psix
    # e=a*t*psi, evaluated at t=1, so e_t=a*psi.
    r = e + u * ex + ux * e + e * ex - nu * amplitude * psixx

    def l2(a):
        return float(np.sqrt(w @ a**2))

    relative_field = l2(e) / l2(u)
    relative_slope = l2(ex) / l2(ux)
    return dict(
        nu=nu,
        delta=delta,
        amplitude=amplitude,
        quadrature_per_panel=order,
        relative_field_l2=relative_field,
        relative_slope_l2=relative_slope,
        center_slope_error=amplitude / delta,
        residual_rms_final=l2(r) / np.sqrt(2),
        field_target_passed=relative_field <= 0.003,
        slope_target_passed=relative_slope <= 0.02,
        pde_target_passed=bool(l2(r) / np.sqrt(2) <= 0.003),
    )


rows = [
    calculate(nu, delta, order)
    for nu in (0.1, 0.01)
    for delta in (0.02, 0.01, 0.005, 0.0025)
    for order in (128, 256)
]
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == AUDIT_SHA
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == REFERENCE_SHA
with BASE.with_suffix(".csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
refinement = max(
    abs(rows[i][key] - rows[i + 1][key])
    for i in range(0, len(rows), 2)
    for key in ("relative_field_l2", "relative_slope_l2", "residual_rms_final")
)
assert refinement < 1e-10
examples = [
    r
    for r in rows
    if r["quadrature_per_panel"] == 256
    and r["field_target_passed"]
    and not r["slope_target_passed"]
]
assert len(examples) > 0
assert all(not r["pde_target_passed"] for r in examples)
summary = dict(
    networks=0,
    optimizer_steps=0,
    rows=len(rows),
    audit_sha256=AUDIT_SHA,
    reference_sha256=REFERENCE_SHA,
    refinement_max=refinement,
    field_pass_slope_fail=examples,
    scope="Explicit smooth perturbations at t=1, exact IC/BC, no PINN. These examples fail PDE too; they do not defeat the combined criteria.",
)
BASE.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
