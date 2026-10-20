"""Compare global and regional sampling errors for one fixed, untrained field."""

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
BASE = Path(__file__).with_name("2026-10-20-unforced-burgers-sampling")


def residual(x, t):
    xx, tt = np.broadcast_arrays(np.asarray(x), np.asarray(t))
    u, ux = np.empty(xx.size), np.empty(xx.size)
    for start in range(0, xx.size, 256):
        stop = start + 256
        u[start:stop] = REFERENCE.reference(xx.flat[start:stop], tt.flat[start:stop], 0.01)
        ux[start:stop] = REFERENCE.reference(
            xx.flat[start:stop], tt.flat[start:stop], 0.01, derivative=True
        )
    u, ux = u.reshape(xx.shape), ux.reshape(xx.shape)
    z = xx / 0.0025
    gaussian = np.exp(-z * z / 2)
    g = z * gaussian
    gx = (1 - z * z) * gaussian / 0.0025
    gxx = (z**3 - 3 * z) * gaussian / 0.0025**2
    psi = g * (1 - xx**2)
    psix = gx * (1 - xx**2) - 2 * xx * g
    psixx = gxx * (1 - xx**2) - 4 * xx * gx - 2 * g
    e, ex = 0.01 * tt * psi, 0.01 * tt * psix
    return 0.01 * psi + u * ex + ux * e + e * ex - 0.01**2 * tt * psixx


def integral(region, order, nt):
    lower, upper, start = (-1, 1, 0) if region == "global" else (-0.1, 0.1, 0.15)
    breaks = (lower, -0.02, 0, 0.02, upper)
    nodes, weights = roots_legendre(order)
    x = np.concatenate([(a + b) / 2 + (b - a) / 2 * nodes for a, b in zip(breaks, breaks[1:])])
    w = np.concatenate([(b - a) / 2 * weights for a, b in zip(breaks, breaks[1:])])
    t = np.linspace(start, 1, nt)
    r = residual(x[:, None], t[None, :])
    volume = (upper - lower) * (1 - start)
    second = float(np.trapz(w @ r**2, t) / volume)
    fourth = float(np.trapz(w @ r**4, t) / volume)
    return dict(rms=float(np.sqrt(second)), shape_factor=fourth / second**2)


references = {
    region: [integral(region, order, nt) for order, nt in ((128, 101), (256, 201))]
    for region in ("global", "regional")
}
rows = []
for seed in range(20261013, 20261045):
    unit = np.random.default_rng(seed).uniform(size=(2048, 2))
    for region in ("global", "regional"):
        x = 2 * unit[:, 0] - 1 if region == "global" else 0.2 * unit[:, 0] - 0.1
        t = unit[:, 1] if region == "global" else 0.15 + 0.85 * unit[:, 1]
        estimate = float(np.sqrt(np.mean(residual(x, t) ** 2)))
        ref = references[region][-1]["rms"]
        rows.append(
            dict(
                region=region,
                seed=seed,
                points=len(unit),
                estimate=estimate,
                reference_rms=ref,
                relative_error=(estimate - ref) / ref,
            )
        )
assert len(rows) == 64
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == AUDIT_SHA
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == REFERENCE_SHA
with BASE.with_suffix(".csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
statistics = {}
for region in ("global", "regional"):
    errors = np.array([r["relative_error"] for r in rows if r["region"] == region])
    statistics[region] = dict(
        reference_rms=references[region][-1]["rms"],
        relative_quadrature_change=abs(
            references[region][0]["rms"] / references[region][1]["rms"] - 1
        ),
        shape_factor=references[region][-1]["shape_factor"],
        shape_factor_relative_refinement=abs(
            references[region][0]["shape_factor"] / references[region][1]["shape_factor"] - 1
        ),
        squared_rms_relative_sd_from_moments=float(
            np.sqrt((references[region][-1]["shape_factor"] - 1) / 2048)
        ),
        squared_rms_relative_sd_observed=float(np.std((1 + errors) ** 2 - 1, ddof=1)),
        relative_error_min=float(errors.min()),
        relative_error_max=float(errors.max()),
        relative_error_median=float(np.median(errors)),
        absolute_relative_error_median=float(np.median(abs(errors))),
        absolute_relative_error_p90=float(np.quantile(abs(errors), 0.9)),
    )
summary = dict(
    networks=0,
    optimizer_steps=0,
    rows=len(rows),
    audit_sha256=AUDIT_SHA,
    reference_sha256=REFERENCE_SHA,
    statistics=statistics,
    scope="Paired uniform draws for two different integration domains of one explicit field; not a PINN or continuous residual certificate. No training threshold is calibrated.",
)
BASE.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
