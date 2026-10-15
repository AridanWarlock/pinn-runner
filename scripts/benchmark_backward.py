"""Compare terminal backward targets on identical saved Robin weights and physical points."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinnstorch.models.pinn_module import PINNModule  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from scripts.benchmark_robin_execution import load_model, objective, point_columns  # noqa: E402
from scripts.benchmark_robin_lifting import measure, synchronize  # noqa: E402


def weight_gradients(
    factory: SolverFactory, model: PINNModule, points: torch.Tensor
) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    model.backward(objective(factory, model, points))
    gradients = {
        name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    return gradients


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument(
        "--lifting",
        choices=("recursive", "expanded"),
        nargs="+",
        default=["recursive", "expanded"],
    )
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.report.exists() or min(args.batch_sizes) < 1:
        parser.error("A new report and positive batch sizes are required")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; no fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rows: list[dict[str, Any]] = []
    differences: dict[str, dict[str, float]] = {}
    profiles: list[dict[str, Any]] = []
    for lifting in args.lifting:
        factory, model = load_model(args.trial, "robin_grouped", lifting, args.device)
        generator = torch.Generator(device=args.device).manual_seed(20261015)
        points = torch.rand(
            (max(args.batch_sizes), len(factory.settings.domain.space) + 1),
            generator=generator,
            device=args.device,
        )
        for column, axis in zip(
            point_columns(points),
            (*factory.settings.domain.space.values(), factory.settings.domain.time),
        ):
            column.mul_(axis.interval[1] - axis.interval[0]).add_(axis.interval[0])
        gradients = {}
        # Compare steady executions, not a cold CUDA kernel with a warmed one.
        # Timing also warms up; self-repeatability is checked independently below.
        for mode in ("all", "parameters"):
            model.hparams["backward_mode"] = mode
            for _ in range(3):
                weight_gradients(factory, model, points)
        for mode in ("all", "parameters"):
            model.hparams["backward_mode"] = mode
            gradients[mode] = weight_gradients(factory, model, points)
            repeated = weight_gradients(factory, model, points)
            for name, gradient in gradients[mode].items():
                torch.testing.assert_close(repeated[name], gradient, rtol=0, atol=0)
        if gradients["all"].keys() != gradients["parameters"].keys():
            raise AssertionError("Backward modes reached different parameter leaves")
        differences[lifting] = {}
        for name, control in gradients["all"].items():
            candidate = gradients["parameters"][name]
            torch.testing.assert_close(candidate, control, rtol=0, atol=0)
            differences[lifting][name] = float((candidate - control).abs().max().cpu())
        # Gradients are evidence only; release their memory before measuring peaks.
        del gradients
        for repeat in range(3):
            for size in args.batch_sizes:
                for mode in ("all", "parameters") if repeat % 2 == 0 else ("parameters", "all"):
                    model.hparams["backward_mode"] = mode

                    def action() -> None:
                        model.zero_grad(set_to_none=True)
                        model.backward(objective(factory, model, points[:size]))
                        model.zero_grad(set_to_none=True)

                    rows.append(
                        {
                            "lifting": lifting,
                            "mode": mode,
                            "repeat": repeat,
                            "batch_size": size,
                            **measure(action, args.device),
                        }
                    )
        if args.trace:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if args.device == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            for mode in ("all", "parameters"):
                model.hparams["backward_mode"] = mode
                model.zero_grad(set_to_none=True)
                with torch.profiler.profile(
                    activities=activities, profile_memory=True
                ) as profiler:
                    with torch.profiler.record_function("pinn/physical_objective"):
                        loss = objective(factory, model, points[:128])
                    with torch.profiler.record_function("pinn/terminal_backward"):
                        model.backward(loss)
                    synchronize(args.device)
                model.zero_grad(set_to_none=True)
                profiles.append(
                    {
                        "lifting": lifting,
                        "mode": mode,
                        "batch_size": min(128, len(points)),
                        "events": [
                            {
                                "name": event.key,
                                "count": event.count,
                                "self_cpu_time_us": event.self_cpu_time_total,
                                "self_cuda_time_us": event.self_cuda_time_total,
                                "self_cuda_memory_bytes": event.self_cuda_memory_usage,
                            }
                            for event in profiler.key_averages()
                        ],
                    }
                )
                args.report.parent.mkdir(parents=True, exist_ok=True)
                profiler.export_chrome_trace(
                    str(args.report.with_suffix(f".{lifting}.{mode}.trace.json"))
                )
    report = {
        "contract": 1,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "trial": str(args.trial),
        "checkpoint_sha256": hashlib.sha256(
            (args.trial / "training/checkpoints/final.ckpt").read_bytes()
        ).hexdigest(),
        "functions_sha256": hashlib.sha256(
            (args.trial / "task/functions.py").read_bytes()
        ).hexdigest(),
        "device": args.device,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "tf32": False,
        "max_parameter_gradient_differences": differences,
        "rows": rows,
        "profiles": profiles,
        "limitations": "Fixed weights; physical derivatives retained; no optimizer steps. 3 alternating repeats, 3 warmups and 9 timed passes per cell; profiler excluded. CPU is tool verification only.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps({"report": str(args.report), "gradient_equivalence": differences}), flush=True
    )


if __name__ == "__main__":
    main()
