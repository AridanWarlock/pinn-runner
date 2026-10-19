"""CPU quadrature of sharp temporal bounds and signed heat balance; no training."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "docs/experiments/2026-10-19-conductivity-error-bound"
z, w = np.polynomial.legendre.leggauss(64)
x, weights = (z + 1) / 2, w / 2
rows = []
for horizon in (0.5, 1.0, 2.0):
    t = horizon * x
    residual = np.cos(np.pi * t / (2 * horizon))
    error = 2 * horizon / np.pi * np.sin(np.pi * t / (2 * horizon))
    ratio = math.sqrt(float(np.dot(weights, error**2) / np.dot(weights, residual**2)))
    expected = 2 * horizon / np.pi
    assert math.isclose(ratio, expected, rel_tol=1e-13, abs_tol=1e-14)
    rows.append(
        dict(
            kind="sharp_constant_mode",
            beta="",
            time=horizon,
            observed=ratio,
            expected=expected,
            absolute_error=abs(ratio - expected),
        )
    )
for beta in (0, 1):
    for instant in (0, 0.17, 0.61, 1):
        amplitude, k0 = 0.7, 1 / 8
        residual_mean = amplitude * float(np.dot(weights, x)) - k0 * beta * amplitude * instant / 2
        # Lower and upper x faces: b=-kappa*normal*e_x; y faces vanish.
        lower = k0 * (1 + beta * x / 2) * amplitude * instant
        upper = -k0 * (1 + beta * (1 + x) / 2) * amplitude * instant
        boundary_flux = float(np.dot(weights, lower + upper))
        observed = residual_mean - boundary_flux
        expected = amplitude / 2
        assert math.isclose(observed, expected, rel_tol=1e-13, abs_tol=1e-14)
        rows.append(
            dict(
                kind="signed_mean_balance",
                beta=beta,
                time=instant,
                observed=observed,
                expected=expected,
                absolute_error=abs(observed - expected),
            )
        )
        mean_free_error = amplitude * instant * (x - 0.5)
        bulk = float(
            np.dot(
                weights, (amplitude * x - k0 * beta * amplitude * instant / 2) * mean_free_error
            )
        )
        boundary = float(
            np.dot(weights, lower * (-amplitude * instant / 2) + upper * (amplitude * instant / 2))
        )
        observed_energy = bulk - boundary
        expected_energy = amplitude**2 * instant / 12 + amplitude**2 * instant**2 * k0 * (
            1 + beta / 2
        )
        assert math.isclose(observed_energy, expected_energy, rel_tol=1e-13, abs_tol=1e-14)
        rows.append(
            dict(
                kind="mean_free_energy_identity",
                beta=beta,
                time=instant,
                observed=observed_energy,
                expected=expected_energy,
                absolute_error=abs(observed_energy - expected_energy),
            )
        )
with PREFIX.with_suffix(".csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
reference_global = 13 / 35 * math.sqrt((1 - math.exp(-2)) / 2)
reference_final = 13 / 35 * math.exp(-1)
residual_scale = 0.0029 * math.sqrt(2)
probe_residual = residual_scale * np.cos(np.pi * x / 2)
probe_error = residual_scale * 2 / np.pi * np.sin(np.pi * x / 2)
probe_rms = math.sqrt(float(np.dot(weights, probe_residual**2)))
probe_relative = math.sqrt(float(np.dot(weights, probe_error**2))) / reference_global
probe_final = residual_scale * 2 / np.pi / reference_final
assert probe_rms < 0.003 and probe_relative > 0.002 and probe_final > 0.003
summary = dict(
    residual_only_counterexample=dict(
        pde_rms=probe_rms,
        relative_l2=probe_relative,
        final_relative_l2=probe_final,
        initial_error=0,
        physical_flux=0,
    ),
    kind="analytic_quadrature_diagnostic",
    device="cpu",
    optimizer_steps=0,
    quadrature_nodes=64,
    passed=len(rows),
    max_absolute_error=max(r["absolute_error"] for r in rows),
    reference_global_l2=reference_global,
    reference_final_l2=reference_final,
    continuous_residual_bound_global_relative=(2 / math.pi * 0.003) / reference_global,
    continuous_residual_bound_final_relative=0.003 / reference_final,
    preregistered_global_relative_gate=0.002,
    preregistered_final_relative_gate=0.003,
    trace_constant_upper_bound=math.sqrt(4 / math.pi**2 + 2 * math.sqrt(2) / math.pi),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limitations="Exact continuous norm assumptions; finite PINN holdouts do not certify these norms. No training or GPU timing.",
)
PREFIX.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
