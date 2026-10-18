"""Pinned full-step experiment, dispatched only through the shared Windows queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def child(args: argparse.Namespace) -> None:
    import torch

    from benchmarks.cases import CASES
    from benchmarks.fingerprints import tensor_fingerprint
    from benchmarks.optimizer_timing import (
        measure_steps,
        measure_total,
        preserved_state,
        summarize,
    )
    from pinnstorch.models.net.spatial_features import FeatureFCN
    from pinnstorch.runner.checkpoint_io import load_checkpoint
    from pinnstorch.runner.factory import SolverFactory
    from pinnstorch.runner.problem import load_problem
    from pinnstorch.runner.settings import TaskSettings
    from pinnstorch.training.refinement import FixedObjective, LBFGSRefinement
    from research_tools.lbfgs_timing import validate
    from research_tools.schema import digest, inside, read_object
    from scripts.profile_convergence_trial import enable_analytic_features, training_step

    root = Path(r"C:\Projects\pinn-runner")
    manifest = validate(read_object(args.manifest))
    if (
        sys.platform != "win32"
        or Path.cwd() != root
        or args.output.exists()
        or os.environ.get("PINN_PINNED_REVISION") != manifest["revision"]
        or not torch.cuda.is_available()
    ):
        raise ValueError("Pinned Windows CUDA snapshot and fresh output required")
    source = manifest["sources"][args.source_index]
    directory = inside(root, source["trial"])
    weights = inside(directory, source["weights"])
    if (
        digest(weights) != source["weights_sha256"]
        or digest(directory / "config.yaml") != source["config_sha256"]
    ):
        raise ValueError("Saved source differs")
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
        raise ValueError("CUDA precision/runtime differs")
    case = CASES[source["case"]]
    settings = TaskSettings.load(directory / "config.yaml")
    if settings.training.seed != source["seed"]:
        raise ValueError("Saved seed differs")
    factory = SolverFactory(settings, load_problem(case.source), directory, case.source.parent)
    objects = factory.build(predicting=False)
    objects.data.setup("fit")
    model = objects.model.to("cuda").eval()
    model.function_mapping = objects.data.function_mapping
    model.load_state_dict(load_checkpoint(weights, map_location="cuda"))
    assert isinstance(model.net, FeatureFCN)
    intervals = [a.interval for a in settings.domain.space.values()] + [
        settings.domain.time.interval
    ]
    generator = torch.Generator(device="cpu").manual_seed(19)
    unit = torch.rand((1281, 4), generator=generator)
    cpu_points = torch.cat(
        [unit[:, i : i + 1] * (b - a) + a for i, (a, b) in enumerate(intervals)], dim=1
    )
    points_sha = hashlib.sha256(cpu_points.numpy().tobytes()).hexdigest()
    points, holdout = cpu_points[:1024].to("cuda"), cpu_points[1024:].to("cuda")
    objective = FixedObjective(
        {"equations": ([points[:, i : i + 1] for i in range(3)], points[:, -1:], {})}
    )

    def field() -> torch.Tensor:
        with torch.no_grad():
            return model.predict_step(
                ([holdout[:, i : i + 1] for i in range(3)], holdout[:, -1:], {}), 0
            )["u"].detach()

    def stage() -> LBFGSRefinement:
        r = settings.training.refinement
        return LBFGSRefinement(
            model, objective, r.learning_rate, r.history_size, r.tolerance_grad, r.tolerance_change
        )

    initial_state = tensor_fingerprint(model.state_dict())
    initial_field = hashlib.sha256(field().cpu().numpy().tobytes()).hexdigest()
    initial_control = enable_analytic_features(factory, model, points)
    model.net.features = replace(model.net.features, derivative_rule=args.rule)
    warmup = stage()
    with preserved_state(model, warmup.optimizer):
        warmup.run(1)
    if tensor_fingerprint(model.state_dict()) != initial_state or warmup.optimizer.state:
        raise ValueError("Warmup failed to restore source/empty optimizer history")
    reports, states, controls, fields, pdes = {}, {}, {}, {}, {}
    order = ("total", "attributed") if args.repeat % 2 == 0 else ("attributed", "total")
    for mode in order:
        model.net.features = replace(model.net.features, derivative_rule=args.rule)
        measured = stage()
        with preserved_state(model, measured.optimizer):

            def sync() -> None:
                torch.cuda.synchronize(model.device)

            torch.cuda.reset_peak_memory_stats(model.device)
            if mode == "total":
                reports[mode] = measure_total(measured.optimizer, measured.closure, 10, sync)
            else:
                reports[mode] = summarize(
                    measure_steps(measured.optimizer, measured.closure, 10, sync)
                )
            reports[mode]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(model.device)
            reports[mode]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(model.device)
            states[mode] = tensor_fingerprint(model.state_dict())
            fields[mode] = field().cpu()
            controls[mode] = enable_analytic_features(factory, model, holdout)
            model.net.features = replace(model.net.features, derivative_rule="autograd")
            pdes[mode] = float(training_step(factory, model, holdout).sqrt())
    if states["total"] != states["attributed"]:
        raise ValueError("Instrumentation changed trajectory from identical state")
    final_field = fields["total"]
    reference = torch.from_numpy(
        case.reference(
            {
                axis: cpu_points[1024:, i : i + 1].double().numpy()
                for i, axis in enumerate(settings.domain.space)
            },
            cpu_points[1024:, -1:].double().numpy(),
        )
    )
    relative = float((final_field.double() - reference).norm() / reference.norm())
    report = dict(
        source_index=args.source_index,
        repeat=args.repeat,
        rule=args.rule,
        source=source,
        revision=manifest["revision"],
        environment=environment,
        initial_state_sha256=initial_state,
        initial_field_sha256=initial_field,
        points_sha256=points_sha,
        initial_control=initial_control,
        final_control=controls["total"],
        total=reports["total"],
        attributed=reports["attributed"],
        timing_paths_match=True,
        final_state_sha256=states,
        restored_state_sha256=tensor_fingerprint(model.state_dict()),
        restored=tensor_fingerprint(model.state_dict()) == initial_state,
        source_unchanged=digest(weights) == source["weights_sha256"]
        and digest(directory / "config.yaml") == source["config_sha256"],
        field=final_field.reshape(-1).tolist(),
        pde_rmse=pdes["total"],
        relative_l2=relative,
        settings=dict(
            learning_rate=settings.training.refinement.learning_rate,
            history_size=settings.training.refinement.history_size,
            tolerance_grad=settings.training.refinement.tolerance_grad,
            tolerance_change=settings.training.refinement.tolerance_change,
            max_iter=1,
            line_search_fn="strong_wolfe",
        ),
        gpu=torch.cuda.get_device_name(),
    )
    args.output.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-index", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--rule", choices=("autograd", "analytic"))
    args = parser.parse_args()
    if args.source_index is not None:
        child(args)
    else:
        from research_tools.lbfgs_timing import run
        from research_tools.queue import registered_manifest, verify_snapshot

        verify_snapshot(args.manifest.parent)
        run(
            Path(r"C:\Projects\pinn-runner"),
            ROOT,
            args.output,
            registered_manifest(args.manifest.parent),
        )


if __name__ == "__main__":
    main()
