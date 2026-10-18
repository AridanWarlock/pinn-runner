"""Synchronized frozen CUDA cost measurements; no profiler and no optimizer updates."""

from __future__ import annotations

import time

PROCESS_START = time.perf_counter()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--rule", choices=("autograd", "analytic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(r"C:\Projects\pinn-runner")
    if sys.platform != "win32" or Path.cwd() != root or args.output.exists():
        parser.error("A fresh output and the guarded Windows CUDA root are required")
    import torch

    from benchmarks.cases import CASES
    from benchmarks.fingerprints import tensor_fingerprint
    from pinnstorch.models.net.spatial_features import FeatureFCN
    from pinnstorch.runner.checkpoint_io import load_checkpoint
    from pinnstorch.runner.factory import SolverFactory
    from pinnstorch.runner.problem import load_problem
    from pinnstorch.runner.settings import TaskSettings
    from research_tools.frozen_timing import validate
    from research_tools.schema import digest, inside, read_object
    from scripts.profile_convergence_trial import enable_analytic_features, training_step

    manifest = validate(read_object(args.manifest))
    if os.environ.get("PINN_PINNED_REVISION") != manifest["revision"]:
        raise ValueError("Pinned source differs")
    source = manifest["sources"][args.source_index]
    directory = inside(root, source["trial"])
    weights = inside(directory, source["weights"])
    if (
        digest(weights) != source["weights_sha256"]
        or digest(directory / "config.yaml") != source["config_sha256"]
    ):
        raise ValueError("Source differs")
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    environment = dict(
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        default_dtype=str(torch.get_default_dtype()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
    )
    if environment != manifest["environment_requirements"]:
        raise ValueError("Runtime differs")
    settings = TaskSettings.load(directory / "config.yaml")
    case = CASES[source["case"]]
    factory = SolverFactory(settings, load_problem(case.source), directory, case.source.parent)
    model = factory.build(predicting=True).model.to("cuda").eval()
    model.load_state_dict(load_checkpoint(weights, map_location="cuda"))
    if not isinstance(model.net, FeatureFCN):
        raise ValueError("Feature network required")
    model.net.features = replace(model.net.features, derivative_rule=args.rule)
    state = tensor_fingerprint(model.state_dict())
    generator = torch.Generator(device="cpu").manual_seed(19)
    unit = torch.rand(
        (manifest["prediction_points"], len(settings.domain.space) + 1), generator=generator
    )
    intervals = [a.interval for a in settings.domain.space.values()] + [
        settings.domain.time.interval
    ]
    cpu_points = torch.cat(
        [unit[:, i : i + 1] * (b - a) + a for i, (a, b) in enumerate(intervals)], dim=1
    )
    points_sha = hashlib.sha256(cpu_points.numpy().tobytes()).hexdigest()
    points = cpu_points.to("cuda")

    def predict(batch_size: int, keep: bool = False) -> torch.Tensor | None:
        values = []
        with torch.no_grad():
            for index, start in enumerate(range(0, len(points), batch_size)):
                batch = points[start : start + batch_size]
                output = model.predict_step(
                    ([batch[:, i : i + 1] for i in range(batch.shape[1] - 1)], batch[:, -1:], {}),
                    index,
                )["u"]
                if keep:
                    values.append(output)
        return torch.cat(values) if keep else None

    torch.cuda.synchronize()
    cold_start = time.perf_counter()
    first = predict(2048, True)
    torch.cuda.synchronize()
    cold_prediction = time.perf_counter() - cold_start
    cold_ready = time.perf_counter() - PROCESS_START
    assert first is not None
    prediction_sha = hashlib.sha256(first.detach().cpu().numpy().tobytes()).hexdigest()
    for batch_size in (128, 2048):
        actual = predict(batch_size, True)
        assert actual is not None
        torch.testing.assert_close(actual, first, rtol=2e-4, atol=5e-6)
    control = enable_analytic_features(factory, model, points[: manifest["points"]])
    model.net.features = replace(model.net.features, derivative_rule=args.rule)

    def measure(operation: Callable[[], Any], seconds: float) -> dict[str, Any]:
        for _ in range(3):
            operation()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(manifest["blocks"]):
            torch.cuda.synchronize()
            started = time.perf_counter()
            calls = 0
            while True:
                operation()
                torch.cuda.synchronize()
                calls += 1
                elapsed = time.perf_counter() - started
                if elapsed >= seconds:
                    break
            samples.append(dict(seconds=elapsed, calls=calls))
        return dict(
            samples=samples,
            median_seconds=statistics.median(s["seconds"] / s["calls"] for s in samples),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )

    objective = measure(
        lambda: training_step(factory, model, points[: manifest["points"]]),
        manifest["block_seconds"],
    )
    prediction = {}
    order = (128, 2048) if (args.source_index + args.repeat) % 2 == 0 else (2048, 128)
    for batch_size in order:
        prediction[str(batch_size)] = measure(
            lambda: predict(batch_size), manifest["prediction_block_seconds"]
        )
    model.zero_grad(set_to_none=True)
    unchanged = tensor_fingerprint(model.state_dict()) == state
    source_unchanged = (
        digest(weights) == source["weights_sha256"]
        and digest(directory / "config.yaml") == source["config_sha256"]
    )
    if not unchanged or not source_unchanged:
        raise ValueError("Frozen source changed")
    report = dict(
        source_index=args.source_index,
        repeat=args.repeat,
        rule=args.rule,
        source=source,
        revision=manifest["revision"],
        environment=environment,
        state_sha256=state,
        points_sha256=points_sha,
        prediction_sha256=prediction_sha,
        weights_unchanged=unchanged,
        source_unchanged=source_unchanged,
        profiled=False,
        independent_autograd_control=control,
        cold_ready_seconds=cold_ready,
        cold_first_prediction_seconds=cold_prediction,
        objective=objective,
        prediction=prediction,
        points=manifest["points"],
        prediction_points=manifest["prediction_points"],
        batch_sizes=[128, 2048],
        last_batch_points=1,
        threads=6,
        gpu=torch.cuda.get_device_name(),
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
