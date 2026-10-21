"""Совместимость второго порядка и ранний поток, без обучения."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.quadratic_reference import reference_fields  # noqa: E402

OUT = Path(__file__).resolve().parent
rows = []
for b in [1.0, 1.5, 2.0]:
    for a in [0.05, 0.1, 0.15]:
        x = torch.tensor([[0.0], [1.0]], dtype=torch.float64, requires_grad=True)
        k = a * (1 + x / b) ** 2

        def differentiate(value):
            return torch.autograd.grad(value.sum(), x, create_graph=True)[0]

        f = torch.sin(torch.pi * x) ** 3
        first = differentiate(k * differentiate(f))
        second = differentiate(k * differentiate(first)).detach().numpy()[:, 0]
        expected = np.array(
            [48 * a * a * np.pi**3 / b, -48 * a * a * np.pi**3 * (1 + 1 / b) ** 3 / b]
        )
        np.testing.assert_allclose(second, expected, atol=1e-10, rtol=1e-13)
        times = np.array([1e-3 / 4**i for i in range(5)])
        fields = reference_fields(
            np.array([0.0, 1.0])[:, None], times[None, :], a=a, b=b, modes=4096
        )
        initial = np.array([6 * a * np.pi**3, 6 * a * (1 + 1 / b) ** 2 * np.pi**3])
        coefficient = np.array(
            [64 * np.pi**2.5 * a**1.5 / b, -64 * np.pi**2.5 * a**1.5 * (1 + 1 / b) ** 2 / b]
        )
        observed = np.stack([fields.dx[0] / times, -fields.dx[1] / times])
        for side in [0, 1]:
            previous = None
            for j, time in enumerate(times):
                estimate = (observed[side, j] - initial[side]) / np.sqrt(time)
                richardson = None if previous is None else 2 * estimate - previous
                previous = estimate
                rows.append(
                    dict(
                        a=a,
                        b=b,
                        side="left" if side == 0 else "right",
                        time=float(time),
                        raw_boundary=float(observed[side, j]),
                        initial_limit=float(initial[side]),
                        sqrt_coefficient=float(coefficient[side]),
                        sqrt_estimate=float(estimate),
                        richardson_estimate=None if richardson is None else float(richardson),
                        richardson_relative_error=None
                        if richardson is None
                        else float(abs(richardson - coefficient[side]) / abs(coefficient[side])),
                        relative_coefficient_error=float(
                            abs(estimate - coefficient[side]) / abs(coefficient[side])
                        ),
                        second_compatibility_defect=float(second[side]),
                    )
                )
with (OUT / "2026-10-21-quadratic-corner-audit.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(
    json.dumps(
        {
            "cases": 9,
            "rows": len(rows),
            "modes": 4096,
            "max_relative_sqrt_coefficient_error_at_smallest_time": max(
                r["relative_coefficient_error"] for r in rows if r["time"] == 1e-3 / 4**4
            ),
            "max_richardson_relative_error_at_smallest_time": max(
                r["richardson_relative_error"] for r in rows if r["time"] == 1e-3 / 4**4
            ),
            "max_second_compatibility_defect": max(
                abs(r["second_compatibility_defect"]) for r in rows
            ),
        },
        indent=2,
    )
)
