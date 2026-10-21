"""Чувствительность разности полей на зарегистрированной сетке; без обучения."""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.catalog import QUADRATIC_EVALUATION as EVALUATION  # noqa: E402
from benchmarks.quadratic_reference import reference_fields  # noqa: E402

x = np.linspace(0, 1, 101)[:, None]
t = np.linspace(0, 1, 101)[None, :]
fields = [reference_fields(x, t, a=a, b=b).temperature for a, b in EVALUATION["hypotheses"]]
rows = []
for i, j in EVALUATION["response_pairs"]:
    left, right = EVALUATION["hypotheses"][i], EVALUATION["hypotheses"][j]
    denominator = np.linalg.norm(fields[j] - fields[i])
    assert denominator > 0
    condition = float((np.linalg.norm(fields[i]) + np.linalg.norm(fields[j])) / denominator)
    rows.append(
        dict(
            kind="shape" if left[0] == right[0] else "scale",
            left_a=left[0],
            left_b=left[1],
            right_a=right[0],
            right_b=right[1],
            condition=condition,
            sufficient_equal_field_error_for_response_5pct=0.05 / condition,
        )
    )
with Path(__file__).with_suffix(".csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(
    f"{len(rows)} pairs; condition range "
    f"{min(r['condition'] for r in rows):.6f} to {max(r['condition'] for r in rows):.6f}"
)
