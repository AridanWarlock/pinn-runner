"""Smooth transport error passes residual/data goals but fails independent field goals."""

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

A, B, DELTA, END = 0.000999, 0.002999, 0.01, 0.5


def error(x, t):
    return A + B * (x + t - np.sqrt((x - t) ** 2 + DELTA**2)) / 2


rows = []
for order in (64, 128, 256):
    nodes, weights = leggauss(order)
    x = (nodes[:, None] + 1) / 2
    t = (nodes[None, :] + 1) * END / 2
    spatial_weights = weights / 2
    time_weights = weights * END / 2
    field_error = error(x, t)
    exact = np.sin(np.pi * (x - t))

    def integral(v):
        return float(spatial_weights @ v @ time_weights)

    final_error = error(x, END).reshape(-1)
    final_reference = np.sin(np.pi * (x - END)).reshape(-1)
    relative = np.sqrt(integral(field_error**2) / integral(exact**2))
    final_relative = np.sqrt(
        float(spatial_weights @ final_error**2) / float(spatial_weights @ final_reference**2)
    )
    initial = np.sqrt(float(spatial_weights @ error(x, 0).reshape(-1) ** 2))
    boundary = float(abs(error(0, t)).max())
    # Analytical first derivatives: the square-root terms cancel in et+ex.
    z = (x - t) / np.sqrt((x - t) ** 2 + DELTA**2)
    ex = B * (1 - z) / 2
    et = B * (1 + z) / 2
    residual = ex + et
    assert relative > 0.002 and final_relative > 0.003
    assert initial < 0.001 and boundary < 0.001 and float(residual.max()) < 0.003
    rows.append(
        dict(
            order=order,
            relative_l2=float(relative),
            final_relative_l2=float(final_relative),
            initial_rmse=float(initial),
            boundary_max=boundary,
            pde_rmse=float(np.sqrt(integral(residual**2) / END)),
            residual_constant_max_error=float(abs(residual - B).max()),
        )
    )
x = np.linspace(0, 1, 41)[:, None]
t = np.linspace(0, END, 31)[None, :]
e = error(x, t)
reference = np.sin(np.pi * (x - t))
grid = dict(
    relative_l2=float(np.linalg.norm(e) / np.linalg.norm(reference)),
    final_relative_l2=float(np.linalg.norm(e[:, -1]) / np.linalg.norm(reference[:, -1])),
    worst_time_relative_l2=float(
        np.max(np.linalg.norm(e, axis=0) / np.linalg.norm(reference, axis=0))
    ),
    initial_rmse=float(np.sqrt(np.mean(e[:, 0] ** 2))),
    boundary_max=float(abs(e[0]).max()),
    pde_rmse=B,
)
assert grid["relative_l2"] > 0.002 and grid["final_relative_l2"] > 0.003
tend = END
unsmoothed_final_squared = (
    A**2 + 2 * A * B * (tend - tend**2 / 2) + B**2 * (tend**2 - 2 * tend**3 / 3)
)
unsmoothed_global_squared = (
    A**2 * tend + A * B * (tend**2 - tend**3 / 3) + B**2 * (tend**3 / 3 - tend**4 / 6)
)
lower_bounds = dict(
    final_relative_l2=float((np.sqrt(unsmoothed_final_squared) - B * DELTA / 2) / np.sqrt(0.5)),
    relative_l2=float(
        (np.sqrt(unsmoothed_global_squared) - B * DELTA / 2 * np.sqrt(tend)) / np.sqrt(tend / 2)
    ),
)
assert lower_bounds["final_relative_l2"] > 0.003 and lower_bounds["relative_l2"] > 0.002
result = dict(
    rigorous_norm_lower_bounds=lower_bounds,
    a=A,
    b=B,
    delta=DELTA,
    time_end=END,
    continuous_quadrature=rows,
    registered_evaluation_grid=grid,
    networks=0,
    optimizer_steps=0,
    source="Analytical smooth perturbation, no optimizer or reference teacher",
)
Path(__file__).with_name("2026-10-20-advection-error-counterexample.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result))
