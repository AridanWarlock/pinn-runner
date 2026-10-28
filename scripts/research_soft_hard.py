"""One prospective soft/hard study through the existing immutable Windows queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def idle_gpu() -> dict:
    """Three pre-trial samples; fail rather than start over another compute owner."""
    samples = []
    began = time.perf_counter()
    while len(samples) < 3 and time.perf_counter() - began < 60:
        p = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        owners = [
            int(v.strip())
            for v in p.stdout.splitlines()
            if v.strip().isdigit() and int(v.strip()) != os.getpid()
        ]
        q = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        utilization = int(q.stdout.split(",")[0].strip())
        if owners or utilization > 5:
            samples = []
        else:
            samples.append(q.stdout.strip())
        time.sleep(1)
    if len(samples) != 3:
        raise RuntimeError("GPU did not become idle; next trial was not started")
    return dict(
        samples=samples, other_compute_processes=False, wait_seconds=time.perf_counter() - began
    )


def main() -> None:
    import lightning
    import numpy as np
    import scipy
    import torch

    from benchmarks.fingerprints import tensor_fingerprint
    from benchmarks.soft_hard_talk import (
        SOURCE,
        Objective,
        costs,
        evaluate,
        field,
        make_model,
        samples,
        train,
    )
    from research_tools.schema import digest, read_object
    from research_tools.soft_hard_talk import ORDER, compare, validate
    from scripts.gpu_worker import write_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    m = validate(read_object(args.manifest))
    if (
        sys.platform != "win32"
        or Path.cwd() != Path(r"C:\Projects\pinn-runner")
        or args.output.exists()
        or os.environ.get("PINN_PINNED_REVISION") != m["revision"]
        or not torch.cuda.is_available()
    ):
        raise ValueError("Pinned Windows snapshot, fresh output and CUDA required")
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("highest")
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
    if environment != m["environment_requirements"]:
        raise ValueError(f"Environment differs: {environment}")
    gpu = torch.cuda.get_device_name()
    if "4070 Ti" not in gpu:
        raise ValueError("RTX 4070 Ti required")
    args.output.mkdir(parents=True)
    sources = [
        SOURCE,
        Path(__file__),
        Path(__file__).resolve().parents[1] / "benchmarks/soft_hard_talk.py",
    ]
    source_before = {str(p): digest(p) for p in sources}
    started = time.perf_counter()
    trials = []
    pair_initial = {}
    progress_path = args.output.parent / "progress.json"
    # Admit the expensive hard path before starting any of the six timed trainings.
    first_idle_gate = idle_gpu()
    admission_model = make_model("hard", 19, "cuda")
    admission_points = samples(19)
    admission_objective = Objective(admission_model, "hard", admission_points, "cuda")
    admission_model.backward(admission_objective.loss())
    if any(
        p.grad is None or not torch.isfinite(p.grad).all() for p in admission_model.parameters()
    ):
        raise ValueError("Pre-series hard CUDA gradients failed")
    admission_faces = {}
    for face in ("x_lower", "x_upper", "y_lower", "y_upper"):
        derivative = "robin" if face == "x_upper" else ("x" if face.startswith("x") else "y")
        admission_faces[face] = float(
            np.max(np.abs(field(admission_model, admission_points[face], "cuda", derivative)))
        )
    if max(admission_faces.values()) > 1e-5:
        raise ValueError("Pre-series hard CUDA boundary audit failed")
    write_json(
        args.output / "admission.json",
        dict(
            optimizer_updates=0,
            hard_boundary_max=admission_faces,
            all_parameter_gradients_finite=True,
            first_idle_gate=first_idle_gate,
        ),
    )
    del admission_model, admission_objective
    torch.cuda.empty_cache()
    evaluation_seconds = 0.0
    np.savez_compressed(
        args.output / "evaluation-weights.npz",
        space=np.linspace(0, 1, 81),
        time=np.linspace(0, 0.5, 41),
        wx=np.r_[0.5, np.ones(79), 0.5] / 80,
        wy=np.r_[0.5, np.ones(79), 0.5] / 80,
        wt=np.r_[0.5, np.ones(39), 0.5] / 40,
    )

    def evaluate_trial(index, row):
        nonlocal evaluation_seconds
        evaluation_started = time.perf_counter()
        seed, method = row["seed"], row["method"]
        directory = args.output / f"{index:02d}-{seed}-{method}"
        model = make_model(method, seed, "cuda").eval()
        row["checkpoint_sha256"] = {
            s["path"]: digest(directory / s["path"]) for s in row["snapshots"]
        }
        for snap_index, snapshot in enumerate(row["snapshots"]):
            checkpoint = torch.load(directory / snapshot["path"], map_location="cuda")
            model.load_state_dict(checkpoint["state_dict"])
            snapshot["metrics"] = evaluate(
                model, "cuda", directory, retain=snap_index == len(row["snapshots"]) - 1
            )
            write_json(
                progress_path,
                dict(
                    stage="evaluation",
                    trial_index=index,
                    snapshot=snap_index,
                    trials_completed=index + 1,
                ),
            )
        del model
        torch.cuda.empty_cache()
        evaluation_seconds += time.perf_counter() - evaluation_started

    for index, (seed, method) in enumerate(ORDER):
        if (
            index % 2 == 0
            and m["settings"]["study_wall_cap_seconds"] - (time.perf_counter() - started)
            < m["settings"]["pair_reserve_seconds"]
        ):
            break
        gate = idle_gpu()
        directory = args.output / f"{index:02d}-{seed}-{method}"
        model = make_model(method, seed, "cuda")
        points = samples(seed)
        parameter_sha = tensor_fingerprint(dict(model.named_parameters()))
        sample_sha = (
            __import__("hashlib")
            .sha256(b"".join(points[k].tobytes() for k in sorted(points)))
            .hexdigest()
        )
        identity = (parameter_sha, sample_sha)
        if seed in pair_initial and pair_initial[seed] != identity:
            raise ValueError("Pair initialization or fixed samples differ before training")
        pair_initial[seed] = identity
        # Numerical admission only: no optimizer updates. Explicitly audit hard invariants before each hard trial.
        if method == "hard":
            probe = torch.tensor(points["ic"][:23], device="cuda")
            with torch.no_grad():
                value = model([probe[:, :1], probe[:, 1:2]], probe[:, 2:])["u"]
            target = torch.cos(torch.pi * probe[:, :1]) * torch.cos(torch.pi * probe[:, 1:2])
            if float((value - target).abs().max()) > 1e-6:
                raise ValueError("CUDA hard IC audit failed")
            for face in ("x_lower", "x_upper", "y_lower", "y_upper"):
                derivative = (
                    "robin" if face == "x_upper" else ("x" if face.startswith("x") else "y")
                )
                if np.max(np.abs(field(model, points[face], "cuda", derivative))) > 1e-5:
                    raise ValueError("CUDA hard boundary audit failed")
        objective = Objective(model, method, points, "cuda")
        model.zero_grad(set_to_none=True)
        model.backward(objective.loss())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("CUDA gradient admission failed")
        model.zero_grad(set_to_none=True)
        row = train(
            model,
            method,
            seed,
            points,
            directory,
            "cuda",
            m["settings"],
            lambda v: write_json(
                progress_path, v | dict(trial_index=index, trials_completed=index)
            ),
        )
        row["idle_gate"] = gate
        trials.append(row)
        write_json(args.output / "training-progress.json", dict(manifest=m, trials=trials))
        del model, objective
        torch.cuda.empty_cache()
        evaluate_trial(index, row)
        write_json(args.output / "evaluated-progress.json", dict(manifest=m, trials=trials))
    cost_started = time.perf_counter()
    for index, row in enumerate(trials):
        seed, method = row["seed"], row["method"]
        directory = args.output / f"{index:02d}-{seed}-{method}"
        model = make_model(method, seed, "cuda").eval()
        before = row["checkpoint_sha256"]
        adam_end = next(s for s in row["snapshots"] if s["stage"] == "adam_end")
        checkpoint = torch.load(directory / adam_end["path"], map_location="cuda")
        row["costs"] = costs(model, method, samples(seed), checkpoint, "cuda", m["settings"])
        row["scientific_checkpoints_unchanged"] = all(
            digest(directory / name) == sha for name, sha in before.items()
        )
        del model, checkpoint
        torch.cuda.empty_cache()
    artifacts = {
        str(p.relative_to(args.output)).replace("\\", "/"): digest(p)
        for p in args.output.rglob("*")
        if p.is_file()
    }
    evidence = dict(
        contract=1,
        kind="soft-hard-talk-evidence",
        manifest=m,
        environment=environment,
        gpu=gpu,
        versions=dict(
            python=sys.version,
            lightning=lightning.__version__,
            numpy=np.__version__,
            scipy=scipy.__version__,
        ),
        source_unchanged=all(digest(p) == source_before[str(p)] for p in sources),
        problem_sha256=digest(SOURCE),
        trials=trials,
        artifacts=artifacts,
        evaluation_seconds=evaluation_seconds,
        cost_seconds=time.perf_counter() - cost_started,
        status="completed" if len(trials) == 6 else "partial",
        stop_reason="completed" if len(trials) == 6 else "user_time_cap",
        total_wall_seconds=time.perf_counter() - started,
        evaluation=dict(
            grid=[81, 81, 41],
            weights="tensor composite trapezoidal normalized per axis",
            pde_points=4096,
            boundary_grid=[161, 81],
            reference_used_for_training=False,
        ),
    )
    compare(evidence)
    (args.output / "study.json").write_text(
        json.dumps(evidence, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_json(
        progress_path,
        dict(stage="completed" if len(trials) == 6 else "partial", trials_completed=len(trials)),
    )
    print(json.dumps(compare(evidence)))


if __name__ == "__main__":
    main()
