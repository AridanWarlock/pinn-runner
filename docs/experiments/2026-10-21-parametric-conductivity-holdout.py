"""Freeze additional material queries and check field/gradient reference grids."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "docs/experiments/2026-10-21-parametric-conductivity"
spec = importlib.util.spec_from_file_location("reference", f"{PREFIX}-reference.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    new_pairs = -0.7 + 1.4 * qmc.Sobol(d=2, scramble=True, seed=20261022).random_base2(6)
    pairs_file = Path(f"{PREFIX}-holdout-parameters.csv")
    with pairs_file.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("index", "a", "b"))
        writer.writerows(
            (i, format(a, ".17g"), format(b, ".17g")) for i, (a, b) in enumerate(new_pairs)
        )
    restored = np.array(
        [[float(r["a"]), float(r["b"])] for r in csv.DictReader(pairs_file.open())]
    )
    assert np.array_equal(new_pairs, restored)
    rows = []
    for index, (a, b) in enumerate([*module.PAIRS, *new_pairs.tolist()]):
        x, u, *_ = module.reference(1024, a, b)
        xc, uc, *_ = module.reference(512, a, b)
        derivative = np.gradient(u, x, axis=0, edge_order=2)
        coarse_derivative = np.gradient(uc, xc, axis=0, edge_order=2)
        field_difference = float(np.abs(u[::2] - uc).max())
        gradient_difference = float(np.abs(derivative[::2] - coarse_derivative).max())
        baseline_error = 0.0
        if a == 0 and b == 0:
            exact = np.pi * np.cos(np.pi * x[:, None]) * np.exp(-(np.pi**2) * module.TIMES / 8)
            baseline_error = float(np.abs(derivative - exact).max())
        rows.append(
            dict(
                index=index,
                kind="original" if index < 13 else "sobol",
                a=a,
                b=b,
                field_grid_max_difference=field_difference,
                gradient_grid_max_difference=gradient_difference,
                constant_material_exact_gradient_max_error=baseline_error,
                passed=bool(
                    field_difference <= 2e-5
                    and gradient_difference <= 2e-4
                    and baseline_error <= 2e-5
                ),
            )
        )
    with Path(f"{PREFIX}-holdout-reference.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        protocol_sha256=hashlib.sha256(
            (ROOT / "docs/parametric-conductivity-holdout-protocol-2026-10-21.md").read_bytes()
        ).hexdigest(),
        optimizer_steps=0,
        original_pairs=13,
        additional_pairs=64,
        parameter_csv_sha256=hashlib.sha256(
            pairs_file.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        parameter_csv_hash_mode="lf-normalized",
        maximum_field_grid_difference=max(r["field_grid_max_difference"] for r in rows),
        maximum_gradient_grid_difference=max(r["gradient_grid_max_difference"] for r in rows),
        all_passed=all(r["passed"] for r in rows),
    )
    Path(f"{PREFIX}-holdout.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    assert result["all_passed"], "Preserve every reference outcome"


if __name__ == "__main__":
    main()
