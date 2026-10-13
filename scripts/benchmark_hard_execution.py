"""Compare hard inference backends on the same saved CUDA heat checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.execution import launch  # noqa: E402
from benchmarks.performance import prediction_speed  # noqa: E402
from pinnstorch.runner.artifacts import ArtifactWriter  # noqa: E402
from pinnstorch.runner.execution import SolverExecution  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402
from pinnstorch.training.constraints import HardBoxInitial  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cold-runs", type=int, default=0)
    parser.add_argument("--skip-warm", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This performance experiment requires the remote CUDA machine")
    if args.report.exists():
        raise ValueError("Refusing to overwrite an existing report")
    if args.cold_runs < 0:
        raise ValueError("cold-runs must be nonnegative")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    directory: Path = args.trial / "task"
    checkpoint: Path = (args.trial / "training/checkpoints/final.ckpt").resolve()
    settings = TaskSettings.load(directory / "config.yaml")
    problem = load_problem(directory / "functions.py")
    objects = SolverFactory(settings, problem, directory).build(True)
    SolverExecution(
        settings,
        objects,
        ArtifactWriter(directory, settings, problem),
        logging.getLogger(__name__),
    )._load(checkpoint, False)
    model = objects.model.to("cuda").eval()
    transform = model.output_fn
    if not isinstance(transform, HardBoxInitial):
        raise ValueError("Checkpoint must use HardBoxInitial")
    axes = [torch.linspace(*interval, 17, device="cuda") for interval in transform.bounds] + [
        torch.linspace(*transform.time_interval, 31, device="cuda")
    ]
    points = torch.cartesian_prod(*axes)
    columns = [points[:, i : i + 1] for i in range(points.shape[1])]
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - started
    rows: list[dict[str, Any]] = []
    max_difference = 0.0
    random_difference = 0.0
    # Reverse order in each repeat to expose ordering/thermal bias.
    with torch.no_grad():
        for repetition in range(0 if args.skip_warm else 3):
            for batch_size in [128, 2048]:
                for enabled in [False, True] if repetition % 2 == 0 else [True, False]:
                    model.output_fn = replace(transform, fused_predict=enabled)
                    torch.cuda.reset_peak_memory_stats()
                    metrics = prediction_speed(model, columns[:-1], columns[-1], batch_size)
                    rows.append(
                        {
                            "repeat": repetition,
                            "fused_predict": enabled,
                            **metrics,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        }
                    )
        for start in range(0, len(points), 2048):
            batch = [column[start : start + 2048] for column in columns]
            model.output_fn = transform
            eager = model.forward(batch[:-1], batch[-1])["u"]
            model.output_fn = replace(transform, fused_predict=True)
            candidate = model.forward(batch[:-1], batch[-1])["u"]
            torch.testing.assert_close(candidate, eager, rtol=2e-6, atol=2e-6)
            max_difference = max(max_difference, float((candidate - eager).abs().max()))
        generator = torch.Generator(device="cuda").manual_seed(20261013)
        random_columns: list[torch.Tensor] = [
            torch.rand(4096, 1, generator=generator, device="cuda").mul(right - left).add(left)
            for left, right in [*transform.bounds, transform.time_interval]
        ]
        model.output_fn = transform
        eager = model.forward(random_columns[:-1], random_columns[-1])["u"]
        model.output_fn = replace(transform, fused_predict=True)
        candidate = model.forward(random_columns[:-1], random_columns[-1])["u"]
        torch.testing.assert_close(candidate, eager, rtol=2e-6, atol=2e-6)
        random_difference = float((candidate - eager).abs().max())
    model.output_fn = transform
    cold_rows: list[dict[str, float | str | int]] = []
    for repetition in range(args.cold_runs):
        for backend in (
            ["eager", "fused_predict"] if repetition % 2 == 0 else ["fused_predict", "eager"]
        ):
            trial = args.report.parent / (args.report.stem + f"-cold-{repetition}-{backend}")
            input_dir, output_dir = trial / "task", trial / "prediction"
            input_dir.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(directory / "functions.py", input_dir / "functions.py")
            config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
            config["training"]["hard_execution"] = backend
            (input_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            seconds = launch(input_dir, output_dir, trial / "runner.log", checkpoint)
            cold_rows.append({"repeat": repetition, "backend": backend, "seconds": seconds})
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "tf32": False,
        "dtype": "float32",
        "setup_seconds": setup_seconds,
        "checkpoint_seed": settings.training.seed,
        "max_field_difference": max_difference,
        "random_holdout_max_difference": random_difference,
        "rows": rows,
        "cold_rows": cold_rows,
        "limitations": "Inference experiment only; no training-speedup claim. Cold child runner excludes parent setup.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
