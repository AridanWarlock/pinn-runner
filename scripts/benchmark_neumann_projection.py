"""Confirm Neumann error groups on independent Gauss quadratures using saved weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.cases import CASES, Array, BoxNeumannHeatCase  # noqa: E402
from benchmarks.neumann_projection import project_neumann_error  # noqa: E402
from scripts.benchmark_robin_execution import load_model  # noqa: E402
from scripts.benchmark_robin_lifting import synchronize  # noqa: E402


def analyze(trial: Path, counts: list[int], time_points: int, device: str) -> dict[str, Any]:
    original = json.loads((trial / "result.json").read_text())
    case = CASES[original["case"]]
    if not isinstance(case, BoxNeumannHeatCase) or any(
        case.mode_for(axis).alpha != 0 for axis in "xyz"[: case.dimensions]
    ):
        raise ValueError("Requires a documented Neumann box heat benchmark")
    source = trial / "task/functions.py"
    if source.read_text(encoding="utf-8") != case.source.read_text(encoding="utf-8"):
        raise ValueError("Benchmark source differs from documented problem")
    factory, model = load_model(
        trial, "robin_grouped", original.get("hard_lifting", "recursive"), device
    )
    bounds = tuple(axis.interval for axis in factory.settings.domain.space.values())
    if any(
        not np.allclose(bound, case.grid.spatial_bounds(), rtol=0, atol=1e-12) for bound in bounds
    ):
        raise ValueError("Domain differs from documented analytical reference")
    if time_points < 2 or not counts or min(counts) < 2:
        raise ValueError("At least two time and quadrature points required")
    times = np.linspace(*factory.settings.domain.time.interval, time_points)
    documented_time = case.config("cpu", 1, original["seed"])["domain"]["time"]["interval"]
    if not np.allclose(factory.settings.domain.time.interval, documented_time, rtol=0, atol=1e-12):
        raise ValueError("Time interval differs from documented analytical reference")
    reference_rate = 0.1 * sum(
        case.mode_for(axis).frequency ** 2 for axis in factory.settings.domain.space
    )
    measurements = []
    for count in counts:
        legendre_rule = cast(Callable[[int], tuple[Array, Array]], np.polynomial.legendre.leggauss)
        z, w = legendre_rule(count)
        nodes = tuple(lower + (z + 1) * (upper - lower) / 2 for lower, upper in bounds)
        weights = tuple(w * (upper - lower) / 2 for lower, upper in bounds)
        mesh = np.meshgrid(*nodes, indexing="ij")
        spatial = {name: x.reshape(-1, 1) for name, x in zip(factory.settings.domain.space, mesh)}
        batch_mesh = np.meshgrid(*nodes, times, indexing="ij")
        features = [
            torch.tensor(x.reshape(-1, 1), device=device, dtype=torch.float32) for x in batch_mesh
        ]
        synchronize(device)
        started = time.perf_counter()
        predictions = []
        for index, start in enumerate(range(0, len(features[0]), 2048)):
            outputs = model.predict_step(
                (
                    [x[start : start + 2048] for x in features[:-1]],
                    features[-1][start : start + 2048],
                    {},
                ),
                index,
            )
            predictions.append(outputs["u"].detach().cpu().numpy().astype(np.float64))
        synchronize(device)
        prediction_seconds = time.perf_counter() - started
        values = np.concatenate(predictions).reshape(-1, time_points)
        error = values - case.reference(spatial, times.reshape(1, -1))
        energy = project_neumann_error(nodes, weights, bounds, error, reference_rate)
        rows = [
            {
                "time": float(instant),
                "error_energy": float(energy.total[i]),
                "error_l2": float(np.sqrt(energy.total[i])),
                **{
                    key + "_fraction": float(value[i] / energy.total[i])
                    if energy.total[i] > 0
                    else None
                    for key, value in energy.groups.items()
                },
            }
            for i, instant in enumerate(times)
        ]
        measurements.append(
            {
                "points_per_axis": count,
                "points": len(features[0]),
                "prediction_seconds": prediction_seconds,
                "prediction_points_per_second": len(features[0]) / prediction_seconds,
                "gram_defect": energy.gram_defect,
                "normalization_defect": energy.normalization_defect,
                "energy_reconstruction_max_error": energy.reconstruction_max_error,
                "mode_indices": energy.indices,
                "rows": rows,
            }
        )
    return {
        "contract": 1,
        "case": case.name,
        "seed": original["seed"],
        "device": device,
        "torch": torch.__version__,
        "trial": str(trial),
        "checkpoint_sha256": hashlib.sha256(
            (trial / "training/checkpoints/final.ckpt").read_bytes()
        ).hexdigest(),
        "reference_decay_rate": reference_rate,
        "measurements": measurements,
        "limitations": "Independent Gauss quadratures; normal coordinate derivatives in the hard field remain active. Inference timing includes transfer to CPU for analysis, not warm deployment latency. No training or PDE-loss change. Near-orthogonality and reconstruction checked; quadrature is an estimate, not a certified continuous error bound. Query coordinates use FP32.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--points", type=int, nargs="+", default=[13, 17])
    parser.add_argument("--time-points", type=int, default=11)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if min(args.points) < 2 or args.time_points < 2 or len(set(args.points)) != len(args.points):
        parser.error("Unique quadrature counts and at least two nodes/times required")
    if args.report.exists():
        parser.error("New report path required")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; no fallback")
    if args.device == "cpu" and (max(args.points) > 5 or args.time_points > 3):
        parser.error("CPU is for small tool checks; numerical acceptance runs on remote CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = analyze(args.trial, args.points, args.time_points, args.device)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "final": [
                    {"points": m["points_per_axis"], **m["rows"][-1]}
                    for m in report["measurements"]
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
