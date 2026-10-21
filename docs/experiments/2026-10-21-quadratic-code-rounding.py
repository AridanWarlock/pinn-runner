"""Отдельный контроль ошибки кода и вычисления коэффициента, без обучения."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.problems.heat_quadratic_shapes.functions import conductivity  # noqa: E402
from pinnstorch.api import Coordinates  # noqa: E402

OUT = Path(__file__).resolve().parent
plan = json.loads((OUT / "2026-10-21-quadratic-shape-plan.json").read_text())
codes = plan["training_rows"] + plan["unseen_rows"]
assert len(set(map(tuple, codes))) == 24
x = np.linspace(0, 1, 1001)
rows = []
for a, b in codes:
    af, bf = float(np.float32(a)), float(np.float32(b))
    assert af >= 0.05 and bf >= 1
    eta = 20 * abs(af - a) + abs(bf - b)
    expected = a * (1 + x / b) ** 2
    rounded = af * (1 + x / bf) ** 2
    observed_eta = float(np.max(abs(np.log(rounded) - np.log(expected))))
    assert observed_eta <= eta + 1e-15
    coordinates = Coordinates(
        spatial={"x": torch.tensor(x[:, None], dtype=torch.float32)},
        time=torch.zeros((len(x), 1), dtype=torch.float32),
        parameters={
            "a": torch.full((len(x), 1), a, dtype=torch.float32),
            "b": torch.full((len(x), 1), b, dtype=torch.float32),
        },
    )
    actual = conductivity(coordinates).detach().numpy()[:, 0].astype(np.float64)
    public_error = float(np.max(abs(actual - expected) / expected))
    assert public_error < 1e-6
    rows.append(
        {
            "a": a,
            "b": b,
            "a_fp32": af,
            "b_fp32": bf,
            "parameter_log_error_uniform_bound": eta,
            "parameter_log_error_observed_on_grid": observed_eta,
            "solution_l2_over_initial_l2_uniform_bound": float(np.sqrt(2) * np.sinh(eta / 2)),
            "public_fp32_coefficient_max_relative_on_grid": public_error,
        }
    )
with (OUT / "2026-10-21-quadratic-code-rounding.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(
    json.dumps(
        {
            "codes": len(rows),
            "points_per_code": len(x),
            "parameter_eta_max": max(r["parameter_log_error_uniform_bound"] for r in rows),
            "solution_normalized_bound_max": max(
                r["solution_l2_over_initial_l2_uniform_bound"] for r in rows
            ),
            "public_value_relative_max_on_grid": max(
                r["public_fp32_coefficient_max_relative_on_grid"] for r in rows
            ),
        },
        indent=2,
    )
)
