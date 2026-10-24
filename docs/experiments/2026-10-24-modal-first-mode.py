"""Первая мода сохраняется точно для произвольных весов; CPU, без обучения."""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.fingerprints import tensor_fingerprint  # noqa: E402
from benchmarks.quadratic_modal_prototype import QuadraticModalPrototype  # noqa: E402

roots, w = leggauss(256)
xi = (roots + 1) / 2
w = w / 2
rows = []
for seed in (11, 29):
    torch.manual_seed(seed)
    model = QuadraticModalPrototype().double().eval()
    before = tensor_fingerprint(model.state_dict())
    for b in (1.0, 1.4, 2.0):
        length = np.log1p(1 / b)
        r = np.exp(length * xi)
        x = b * np.expm1(length * xi)
        measure = b * length * r * w
        phi = r**-0.5 * np.sin(np.pi * xi)
        initial_projection = float(measure @ (np.sin(np.pi * x) ** 3 * phi))
        for a in (0.05, 0.15):
            decay = a / b**2 * ((np.pi / length) ** 2 + 0.25)
            for time in (0.0, 0.25, 1.0):
                xx = torch.tensor(x[:, None], dtype=torch.float64)
                with torch.no_grad():
                    actual = (
                        model(
                            [xx, torch.full_like(xx, a), torch.full_like(xx, b)],
                            torch.full_like(xx, time),
                        )["u"]
                        .numpy()
                        .ravel()
                    )
                projection = float(measure @ (actual * phi))
                expected = float(initial_projection * np.exp(-decay * time))
                assert abs(projection - expected) < 1e-11
                rows.append(
                    dict(
                        seed=seed,
                        a=a,
                        b=b,
                        time=time,
                        projection=projection,
                        expected=expected,
                        absolute_error=abs(projection - expected),
                    )
                )
    assert tensor_fingerprint(model.state_dict()) == before
with Path(__file__).with_suffix(".csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(
    {
        "checks": len(rows),
        "maximum_projection_error": max(row["absolute_error"] for row in rows),
        "device": "cpu",
        "training": False,
    }
)
