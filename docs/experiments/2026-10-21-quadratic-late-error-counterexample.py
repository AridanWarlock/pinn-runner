"""Аналитическое поле проходит глобальные цели, но нарушает позднюю точность."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.catalog import QUADRATIC_EVALUATION  # noqa: E402
from benchmarks.quadratic_reference import reference_fields  # noqa: E402

OUT = Path(__file__).resolve().parent
AMPLITUDE = 0.018
x = np.linspace(0, 1, 101)[:, None]
t = np.linspace(0, 1, 101)[None, :]
probes = np.random.default_rng(20261021).uniform(size=(1024, 2))


def bump(value):
    return float(np.exp(1 - 1 / (1 - value**2))) if abs(value) < 1 else 0.0


def mode(x, a, b):
    length = np.log1p(1 / b)
    nu = np.pi / length
    norm = np.sqrt(b * length / 2)
    r = 1 + x / b
    phi = r**-0.5 * np.sin(nu * np.log1p(x / b)) / norm
    dx = (
        r**-1.5
        / b
        * (nu * np.cos(nu * np.log1p(x / b)) - 0.5 * np.sin(nu * np.log1p(x / b)))
        / norm
    )
    decay = a / b**2 * (nu**2 + 0.25)
    return phi, dx, decay


def temporal(t, decay):
    norm = np.sqrt(-np.expm1(-2 * decay) / (2 * decay))
    error = AMPLITUDE * np.exp(-decay) * np.sinh(decay * t) / (decay * norm)
    residual = AMPLITUDE * np.exp(-decay * (1 - t)) / norm
    return error, residual


def relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


rows, actual, references = [], [], []
for (a, b), split in zip(QUADRATIC_EVALUATION["hypotheses"], QUADRATIC_EVALUATION["splits"]):
    factor = bump((0.15 - a) / 0.02) * bump((b - 1) / 0.1)
    ref = reference_fields(x, t, a=a, b=b)
    phi, dx, decay = mode(x, a, b)
    delta, _ = temporal(t, decay)
    prediction = ref.temperature + factor * phi * delta
    derivative = ref.dx + factor * dx * delta
    k = a * (1 + x / b) ** 2
    sample_phi, _, _ = mode(probes[:, :1], a, b)
    _, sample_g = temporal(probes[:, 1:], decay)
    rms = float(np.sqrt(np.mean((factor * sample_phi * sample_g) ** 2)))
    times = np.linalg.norm(prediction - ref.temperature, axis=0) / np.linalg.norm(
        ref.temperature, axis=0
    )
    row = dict(
        a=a,
        b=b,
        split=split,
        bump=factor,
        relative_l2=relative(prediction, ref.temperature),
        final_relative_l2=float(times[-1]),
        worst_time_relative_l2=float(times.max()),
        dx_relative_l2=relative(derivative, ref.dx),
        flux_relative_l2=relative(-k * derivative, -k * ref.dx),
        pde_rmse=rms,
        initial_rmse=float(np.sqrt(np.mean((prediction[:, 0] - ref.temperature[:, 0]) ** 2))),
        boundary_max=float(abs(prediction[[0, -1]]).max()),
    )
    assert (
        row["relative_l2"] <= 0.02
        and row["dx_relative_l2"] <= 0.05
        and row["flux_relative_l2"] <= 0.05
        and rms <= 0.02
    )
    assert row["initial_rmse"] <= 1e-6 and row["boundary_max"] <= 1e-6
    rows.append(row)
    actual.append(prediction)
    references.append(ref.temperature)
responses = [
    relative(actual[j] - actual[i], references[j] - references[i])
    for i, j in QUADRATIC_EVALUATION["response_pairs"]
]
assert max(responses) == 0
assert max(row["final_relative_l2"] for row in rows) > 0.03
# Независимое автоматическое дифференцирование возмущения в худшем углу.
a, b = 0.15, 1.0
xt = torch.tensor(probes[:71, :1], dtype=torch.float64, requires_grad=True)
tt = torch.tensor(probes[:71, 1:], dtype=torch.float64, requires_grad=True)
length = np.log1p(1 / b)
nu = np.pi / length
decay = a / b**2 * (nu**2 + 0.25)
norm = np.sqrt(-np.expm1(-2 * decay) / (2 * decay))
psi = (1 + xt / b) ** -0.5 * torch.sin(nu * torch.log1p(xt / b)) / np.sqrt(b * length / 2)
error = psi * AMPLITUDE * np.exp(-decay) * torch.sinh(decay * tt) / (decay * norm)
dt = torch.autograd.grad(error.sum(), tt, create_graph=True)[0]
dx = torch.autograd.grad(error.sum(), xt, create_graph=True)[0]
div = torch.autograd.grad((a * (1 + xt / b) ** 2 * dx).sum(), xt)[0]
expected = psi * AMPLITUDE * torch.exp(-decay * (1 - tt)) / norm
np.testing.assert_allclose(
    (dt - div).detach().numpy(), expected.detach().numpy(), atol=1e-13, rtol=1e-12
)
# Пространственно-временная L2-норма невязки известна точно, сверяется квадратурой.
roots, weights = leggauss(128)
nodes = (roots + 1) / 2
weights = weights / 2
phi, _, decay = mode(nodes[:, None], a, b)
_, g = temporal(nodes[None, :], decay)
continuous_rms = float(np.sqrt(weights @ (phi * g) ** 2 @ weights))
assert abs(continuous_rms - AMPLITUDE) < 1e-13
with (OUT / "2026-10-21-quadratic-late-error-counterexample.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
summary = dict(
    kind="analytical_counterexample_not_trained_network",
    residual_l2_exact=AMPLITUDE,
    residual_l2_quadrature=continuous_rms,
    coefficient_response_error=max(responses),
    maxima={
        key: max(row[key] for row in rows)
        for key in [
            "relative_l2",
            "final_relative_l2",
            "worst_time_relative_l2",
            "dx_relative_l2",
            "flux_relative_l2",
            "pde_rmse",
            "initial_rmse",
            "boundary_max",
        ]
    },
)
(OUT / "2026-10-21-quadratic-late-error-counterexample.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
