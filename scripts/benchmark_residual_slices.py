"""Check physical PDE residuals on common spatial holdout slices, including t0 and T."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinnstorch.models.pinn_module import PINNModule  # noqa: E402
from scripts.benchmark_robin_execution import load_model  # noqa: E402


def verify_saved_prediction(
    trial: Path, model: PINNModule, spatial_names: tuple[str, ...], fields: tuple[str, ...]
) -> dict[str, float | int]:
    """Reevaluate every saved endpoint layer and common interior rows by their labels."""
    saved = loadmat(trial / "prediction/predictions.mat")
    times = np.asarray(saved["t"]).reshape(-1)
    space_rows = len(np.asarray(saved[spatial_names[0]]).reshape(-1))
    for name in fields:
        if np.asarray(saved[name]).shape != (space_rows, len(times)):
            raise ValueError("Saved field must have shape (spatial rows, time instants)")
    time_endpoints = np.flatnonzero((times == times.min()) | (times == times.max()))
    endpoints = (np.arange(space_rows)[:, None] * len(times) + time_endpoints[None, :]).reshape(-1)
    interior = np.random.default_rng(20261016).choice(
        space_rows * len(times), min(128, space_rows * len(times)), replace=False
    )
    selected = np.unique(np.concatenate((endpoints, interior)))
    maximum = 0.0
    for start in range(0, len(selected), 512):
        indices = selected[start : start + 512]
        columns = [
            torch.tensor(
                np.asarray(saved[name]).reshape(-1, 1)[indices // len(times)],
                dtype=torch.float32,
                device=model.device,
            )
            for name in spatial_names
        ]
        columns.append(
            torch.tensor(
                times[indices % len(times)].reshape(-1, 1),
                dtype=torch.float32,
                device=model.device,
            )
        )
        with torch.no_grad():
            actual = model(columns[:-1], columns[-1])
        for name in fields:
            expected = torch.tensor(
                np.asarray(saved[name]).reshape(-1, 1)[indices],
                dtype=torch.float32,
                device=model.device,
            )
            torch.testing.assert_close(actual[name], expected, atol=3e-6, rtol=3e-5)
            maximum = max(maximum, float((actual[name] - expected).abs().max().cpu()))
    return {
        "saved_prediction_points_checked": len(selected),
        "saved_prediction_endpoint_points_checked": len(endpoints),
        "saved_prediction_max_difference": maximum,
    }


def evaluate(trial: Path, count: int, layers: int) -> dict[str, Any]:
    result = json.loads((trial / "result.json").read_text())
    factory, model = load_model(trial, "robin_grouped", result.get("hard_lifting", "recursive"))
    settings = factory.settings
    generator = np.random.default_rng(20261016)
    points = generator.uniform(size=(count, len(settings.domain.space)))
    for index, axis in enumerate(settings.domain.space.values()):
        points[:, index] = axis.interval[0] + points[:, index] * (
            axis.interval[1] - axis.interval[0]
        )
    instants = np.linspace(*settings.domain.time.interval, layers)
    rows = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for instant in instants:
        spatial = [
            torch.tensor(
                points[:, i : i + 1], dtype=torch.float32, device="cuda", requires_grad=True
            )
            for i in range(points.shape[1])
        ]
        clock = torch.full_like(spatial[0], float(instant), requires_grad=True)
        with torch.set_grad_enabled(True):
            physical = dict(model(spatial, clock))
            residuals = factory.adapter.residuals(physical, *spatial, clock)
            values = (
                torch.cat([residuals[name] for name in factory.problem.equations], dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        rows.append(
            {
                "time": float(instant),
                "pde_rmse": float(np.sqrt(np.mean(values**2))),
                "max_absolute": float(np.max(np.abs(values))),
                "signed_mean": float(np.mean(values)),
            }
        )
    torch.cuda.synchronize()
    checkpoint = torch.load(trial / "training/checkpoints/final.ckpt", map_location="cpu")
    training_times = (
        checkpoint["refinement_state"]["fixed_batches"]["equations"][1].numpy().reshape(-1)
    )
    edges = np.linspace(*settings.domain.time.interval, 11)
    return {
        "trial": str(trial),
        "case": result["case"],
        "seed": result["seed"],
        "sampling_recipe": result["sampling_recipe"],
        "hard_lifting": result.get("hard_lifting", "recursive"),
        "holdout_seed": 20261016,
        "holdout_points_per_layer": count,
        "spatial_points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
        "rows": rows,
        "training_time_bins": np.histogram(training_times, edges)[0].tolist(),
        "training_min_time": float(training_times.min()),
        "diagnostic_seconds": time.perf_counter() - started,
        **verify_saved_prediction(
            trial, model, tuple(settings.domain.space), tuple(factory.problem.fields)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--layers", type=int, default=21)
    args = parser.parse_args()
    if args.report.exists() or args.points < 1 or args.layers < 2:
        parser.error("New output path, positive point count and at least two layers required")
    if not torch.cuda.is_available():
        parser.error("Saved-model numerical slices require remote CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    trials = [path.parent for root in args.roots for path in root.glob("*/result.json")]
    if not trials:
        parser.error("No completed trials found")
    report = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "trials": [evaluate(trial, args.points, args.layers) for trial in trials],
        "limitations": "Same independent uniform spatial points on each physical time slice; includes endpoints. No analytical solution, no optimizer update, no new acceptance threshold. Only exact Robin checkpoints are supported by this tool.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(args.report), "trials": len(trials)}))


if __name__ == "__main__":
    main()
