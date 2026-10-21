"""Точное вынесение масштаба коэффициента в часы; только CPU-эталон."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.catalog import QUADRATIC_TRAINING_ROWS, QUADRATIC_UNSEEN_ROWS  # noqa: E402
from benchmarks.quadratic_reference import reference_fields  # noqa: E402

x = np.linspace(0, 1, 101)[:, None]
t = np.linspace(0, 1, 101)[None, :]
a_reference = 0.15
rows = []
for a, b in QUADRATIC_TRAINING_ROWS + QUADRATIC_UNSEEN_ROWS:
    scale = a / a_reference
    original = reference_fields(x, t, a=a, b=b)
    normalized = reference_fields(x, scale * t, a=a_reference, b=b)
    differences = dict(
        field=float(np.max(abs(original.temperature - normalized.temperature))),
        dx=float(np.max(abs(original.dx - normalized.dx))),
        dxx=float(np.max(abs(original.dxx - normalized.dxx))),
        dt=float(np.max(abs(original.dt - scale * normalized.dt))),
    )
    assert max(differences.values()) < 1e-12, (a, b, differences)
    rows.append(dict(a=a, b=b, clock_scale=scale, **differences))
output = Path(__file__).parent / "2026-10-21-quadratic-clock-audit.csv"
with output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(
    json.dumps(
        dict(codes=len(rows), maxima={key: max(r[key] for r in rows) for key in differences}),
        indent=2,
    )
)
