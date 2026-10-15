"""Inspect slow Neumann error modes in saved MAT predictions; no model or GPU run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.cases import CASES, BoxNeumannHeatCase  # noqa: E402
from benchmarks.cosine_spectrum import cosine_spectrum  # noqa: E402


def analyze(trial: Path) -> dict[str, Any]:
    original = json.loads((trial / "result.json").read_text())
    case = CASES[original["case"]]
    if not isinstance(case, BoxNeumannHeatCase) or any(
        case.mode_for(axis).alpha != 0 for axis in "xyz"[: case.dimensions]
    ):
        raise ValueError("Requires a documented box heat case with Neumann on every face")
    source = trial / "task/functions.py"
    source_text = source.read_text(encoding="utf-8")
    if source_text != case.source.read_text(encoding="utf-8"):
        raise ValueError("Benchmark source differs from the documented analytical problem")
    path = trial / "prediction/predictions.mat"
    saved = loadmat(path)
    spatial = {
        axis: np.asarray(saved[axis], dtype=np.float64) for axis in "xyz"[: case.dimensions]
    }
    for column in spatial.values():
        lower, upper = case.grid.spatial_bounds()
        if not np.isclose(column.min(), lower) or not np.isclose(column.max(), upper):
            raise ValueError("Prediction domain differs from documented reference")
    clock = np.asarray(saved["t"], dtype=np.float64).reshape(-1)
    values = np.asarray(saved["u"], dtype=np.float64)
    expected = case.reference(spatial, clock.reshape(1, -1))
    result = cosine_spectrum(spatial, values - expected)
    rate = 0.1 * sum(case.mode_for(axis).frequency ** 2 for axis in spatial)
    groups = result.grouped_energy(rate)
    rows = []
    for i, time in enumerate(clock):
        total = float(result.sampled_error_energy[i])
        rows.append(
            {
                "time": float(time),
                "error_energy": total,
                "error_l2": float(np.sqrt(total)),
                **{
                    name + "_fraction": float(energy[i] / total) if total > 0 else None
                    for name, energy in groups.items()
                },
            }
        )
    top = np.argsort(result.modal_energy[:, -1])[::-1][:20]
    modes = [
        {
            "index": result.indices[i],
            "decay_rate": float(result.eigenvalues[i]),
            "final_coefficient": float(result.coefficients[i, -1]),
            "final_energy": float(result.modal_energy[i, -1]),
            "energy_fraction": float(result.modal_energy[i, -1] / result.sampled_error_energy[-1])
            if result.sampled_error_energy[-1] > 0
            else None,
        }
        for i in top
    ]
    return {
        "contract": 1,
        "case": case.name,
        "seed": original["seed"],
        "trial": str(trial),
        "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "functions_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "normalized_functions_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "reference_decay_rate": rate,
        "quadrature": "tensor-product trapezoidal",
        "shape": [len(np.unique(x)) for x in spatial.values()],
        "parseval_max_error": result.parseval_max_error,
        "rows": rows,
        "largest_final_modes": modes,
        "limitations": "Full discrete cosine basis on saved uniform grid. Trapezoidal inner product differs from equal-point metrics. Unresolved continuous modes can alias; sampled Parseval identity is not a bound on the continuous solution error. Only documented Neumann heat cases, no training or inference. Original quality gates are unchanged.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("A new report path is required")
    report = analyze(args.trial)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "final": report["rows"][-1],
                "largest_modes": report["largest_final_modes"][:5],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
