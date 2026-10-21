"""Таблица следующего опыта и проверка различимости без обучения."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.quadratic_reference import reference_fields  # noqa: E402

OUT = Path(__file__).resolve().parent


def reverse_binary(n):
    result, weight = 0.0, 0.5
    while n:
        result += (n % 2) * weight
        n //= 2
        weight /= 2
    return result


corners = [[a, b] for a in [0.05, 0.15] for b in [1.0, 2.0]]
training = corners + [[0.05 + 0.1 * (i + 0.5) / 12, 1 + reverse_binary(i + 1)] for i in range(12)]
unseen = [[a, b] for a in [0.075, 0.125] for b in [1.2, 1.4, 1.6, 1.8]]
evaluation = unseen + corners
assert len(set(map(tuple, training))) == 16
assert not set(map(tuple, training)) & set(map(tuple, unseen))
assert not set(row[1] for row in training) & set(row[1] for row in unseen)
x = np.linspace(0, 1, 101)[:, None]
t = np.linspace(0, 1, 101)[None, :]
fields = {tuple(row): reference_fields(x, t, a=row[0], b=row[1]).temperature for row in evaluation}
rows = []
for a, b in evaluation:
    expected = fields[a, b]
    ignored_a = reference_fields(x, t, a=0.1, b=b).temperature
    ignored_b = reference_fields(x, t, a=a, b=1.5).temperature
    analytic = reference_fields(x, t, a=a, b=b)
    flux = -a * (1 + x / b) ** 2 * analytic.dx
    rows.append(
        {
            "a": a,
            "b": b,
            "split": "unseen" if [a, b] in unseen else "training_corner",
            "field_norm": float(np.linalg.norm(expected)),
            "final_field_norm": float(np.linalg.norm(expected[:, -1])),
            "dx_norm": float(np.linalg.norm(analytic.dx)),
            "flux_norm": float(np.linalg.norm(flux)),
            "ignored_a_reference_relative_l2": float(
                np.linalg.norm(ignored_a - expected) / np.linalg.norm(expected)
            ),
            "ignored_b_reference_relative_l2": float(
                np.linalg.norm(ignored_b - expected) / np.linalg.norm(expected)
            ),
        }
    )
pairs = [("shape", [a, 1.2], [a, 1.8]) for a in [0.075, 0.125]]
pairs += [("scale", [0.075, b], [0.125, b]) for b in [1.2, 1.4, 1.6, 1.8]]
response_rows = []
for kind, left, right in pairs:
    response = fields[tuple(right)] - fields[tuple(left)]
    response_rows.append(
        {
            "kind": kind,
            "left": left,
            "right": right,
            "reference_response_norm": float(np.linalg.norm(response)),
            "relative_to_right_field": float(
                np.linalg.norm(response) / np.linalg.norm(fields[tuple(right)])
            ),
            "ignored_component_response_error": 1.0,
        }
    )
plan = {
    "kind": "prospective_plan_not_executable_queue_protocol",
    "training_rows": training,
    "unseen_rows": unseen,
    "training_controls": corners,
    "seeds": [11, 29, 83],
    "adam_steps": 500,
    "lbfgs_external_steps": 500,
    "physical_lhs_points": 4096,
    "evaluation_grid": [101, 101],
    "reference_modes": 64,
    "pde_points_per_hypothesis": 1024,
    "pde_seed": 20261021,
    "targets": {
        "relative_l2": 0.02,
        "final_relative_l2": 0.03,
        "worst_time_relative_l2": 0.03,
        "dx_relative_l2": 0.05,
        "flux_relative_l2": 0.05,
        "pde_rmse": 0.02,
        "initial_rmse": 1e-6,
        "boundary_max": 1e-6,
        "response_relative_l2": 0.05,
    },
    "responses": response_rows,
}
(OUT / "2026-10-21-quadratic-shape-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
with (OUT / "2026-10-21-quadratic-shape-plan-checks.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(
    json.dumps(
        {
            "training": len(training),
            "unseen": len(unseen),
            "controls": len(corners),
            "shape_response_relative_range": [
                min(r["relative_to_right_field"] for r in response_rows if r["kind"] == "shape"),
                max(r["relative_to_right_field"] for r in response_rows if r["kind"] == "shape"),
            ],
            "min_final_field_norm": min(r["final_field_norm"] for r in rows),
            "min_dx_norm": min(r["dx_norm"] for r in rows),
            "min_flux_norm": min(r["flux_norm"] for r in rows),
        },
        indent=2,
    )
)
