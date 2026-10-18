"""One unchanged saved state, five allocator regions; no optimizer updates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from benchmarks.allocation_profile import profile_allocations  # noqa: E402
from benchmarks.cases import CASES  # noqa: E402
from benchmarks.fingerprints import tensor_fingerprint  # noqa: E402
from pinnstorch.models.net.spatial_features import FeatureFCN  # noqa: E402
from pinnstorch.runner.checkpoint_io import load_checkpoint  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402
from research_tools.schema import digest  # noqa: E402
from scripts.profile_convergence_trial import (  # noqa: E402
    enable_analytic_features,
    parameter_gradient,
    training_step,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--rule", choices=("autograd", "analytic"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (
        sys.platform != "win32"
        or Path.cwd() != Path(r"C:\Projects\pinn-runner")
        or args.output.exists()
        or not sys.dont_write_bytecode
    ):
        p.error("Guarded Windows root and fresh output required")
    if not torch.cuda.is_available():
        p.error("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = TaskSettings.load(args.trial / "config.yaml")
    if settings.runtime.device != "cuda":
        p.error("Saved configuration must use CUDA")
    case = CASES["heat_neumann_polynomial2d"]
    factory = SolverFactory(settings, load_problem(case.source), args.trial, case.source.parent)
    model = factory.build(predicting=True).model.to("cuda").eval()
    weights = args.trial / args.weights
    before = digest(weights)
    model.load_state_dict(load_checkpoint(weights, map_location="cuda"))
    initial = tensor_fingerprint(model.state_dict())
    if not isinstance(model.net, FeatureFCN) or model.net.features.kind != "neumann_polynomial":
        raise ValueError("Expected polynomial features")
    unit = torch.rand((1024, 3), generator=torch.Generator().manual_seed(19))
    intervals = [a.interval for a in settings.domain.space.values()] + [
        settings.domain.time.interval
    ]
    points = torch.cat(
        [unit[:, i : i + 1] * (b - a) + a for i, (a, b) in enumerate(intervals)], dim=1
    ).to("cuda")
    independent = enable_analytic_features(factory, model, points)
    model.net.features = replace(model.net.features, derivative_rule=args.rule)
    reference = training_step(factory, model, points)
    gradient = parameter_gradient(model)
    args.output.mkdir(parents=True)
    profiling = profile_allocations(factory, model, points, args.output)
    # The decomposed forward must preserve the ordinary physical-loss gradient.
    torch.testing.assert_close(parameter_gradient(model), gradient, rtol=2e-4, atol=2e-7)
    torch.testing.assert_close(
        torch.tensor(profiling["loss_value"], device=reference.device),
        reference,
        rtol=2e-4,
        atol=1e-8,
    )
    model.zero_grad(set_to_none=True)
    if tensor_fingerprint(model.state_dict()) != initial or digest(weights) != before:
        raise ValueError("Allocation observation changed source weights")
    environment = dict(
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        default_dtype=str(torch.get_default_dtype()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
    )
    report = dict(
        environment=environment,
        gpu=torch.cuda.get_device_name(),
        revision=os.environ["PINN_PINNED_REVISION"],
        rule=args.rule,
        weights_sha256=before,
        state_sha256=initial,
        source_unchanged=True,
        independent_autograd_control=independent,
        decomposed_gradient_passed=True,
        profiling=profiling,
        artifact_sha256={"allocator-trace.json": digest(args.output / "allocator-trace.json")},
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
