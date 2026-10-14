"""Analyse saved benchmark predictions without rerunning training or inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.cases import CASES, BoxNeumannHeatCase  # noqa: E402
from benchmarks.metrics import temporal_errors  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Refusing to overwrite an existing report")
    original = json.loads((args.trial / "result.json").read_text())
    case = CASES[original["case"]]
    if not isinstance(case, BoxNeumannHeatCase):
        raise ValueError("Modal decomposition requires a documented Cartesian operator heat case")
    if (args.trial / "task/functions.py").read_bytes() != case.source.read_bytes():
        raise ValueError("Benchmark source differs from the documented analytical problem")
    saved = loadmat(args.trial / "prediction/predictions.mat")
    spatial = {
        axis: np.asarray(saved[axis], dtype=np.float64) for axis in "xyz"[: case.dimensions]
    }
    time = np.asarray(saved["t"], dtype=np.float64)
    prediction = np.asarray(saved["u"], dtype=np.float64)
    rows = temporal_errors(case, spatial, time, prediction)
    keys = ("mode_relative_l2", "mode_projection_relative_error", "mode_orthogonal_rmse")
    summary = {}
    for key in keys:
        values = [row[key] for row in rows]
        if any(value is None for value in values):
            raise ValueError("Undefined modal diagnostic")
        summary[f"worst_{key}"] = max(float(value) for value in values if value is not None)
        final = rows[-1][key]
        if final is None:
            raise ValueError("Undefined final modal diagnostic")
        summary[f"final_{key}"] = final
    report = {
        "case": case.name,
        "seed": original["seed"],
        "constraints": original["constraint_kind"],
        "execution": original["hard_execution"],
        "prediction_file": str(args.trial / "prediction/predictions.mat"),
        "diagnostic_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "rows": rows,
        "summary": summary,
        "limitations": "Discrete Euclidean projection on the documented mode; aggregate mode_relative_l2 uses the whole error. No training, no inference and no new acceptance gate.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
