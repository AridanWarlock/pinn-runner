"""Compare two reference orders on every registered evaluation node, on CPU."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.burgers import reference  # noqa: E402


def main():
    x, t = np.linspace(-1, 1, 1601), np.linspace(0, 1, 401)
    rows = []
    for viscosity in (0.1, 0.01):
        for derivative in (False, True):
            left, right = [
                reference(x[:, None], t[None, :], viscosity, derivative=derivative, order=order)
                for order in (512, 1024)
            ]
            assert (
                left.shape == (1601, 401) and np.isfinite(left).all() and np.isfinite(right).all()
            )
            difference = np.abs(left - right)
            i, j = np.unravel_index(np.argmax(difference), difference.shape)
            limit = 1e-8 if derivative else 1e-10
            rows.append(
                dict(
                    viscosity=viscosity,
                    quantity="gradient" if derivative else "field",
                    nodes=difference.size,
                    lower_order=512,
                    higher_order=1024,
                    maximum_absolute_difference=float(difference[i, j]),
                    x=float(x[i]),
                    time=float(t[j]),
                    absolute_limit=limit,
                    passed=bool(difference[i, j] <= limit),
                )
            )
    base = Path(__file__).with_name("2026-10-20-unforced-burgers-run-reference-grid")
    assert not base.with_suffix(".csv").exists() and not base.with_suffix(".json").exists()
    with base.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        comparisons=4,
        unique_nodes_per_viscosity=1601 * 401,
        passed=sum(r["passed"] for r in rows),
        device="cpu",
        dtype="float64",
        optimizer_steps=0,
        reference_sha256=hashlib.sha256(
            (ROOT / "benchmarks/burgers.py").read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        protocol_sha256=hashlib.sha256(
            (ROOT / "docs/unforced-burgers-reference-grid-audit-2026-10-20.md").read_bytes()
        ).hexdigest(),
        scope="Full registered-grid order convergence only; same formula, no continuous or interval certificate.",
    )
    base.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
