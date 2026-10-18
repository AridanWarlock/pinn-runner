"""Bounded native CUDA trace of frozen weights, using the existing job queue."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from .schema import Json, digest, inside, object_value, read_object

NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.1.2\target-windows-x64\nsys.exe"
FIELDS = {
    "contract",
    "kind",
    "registration",
    "revision",
    "case",
    "trial",
    "weights",
    "weights_sha256",
    "config_sha256",
    "points",
    "environment_requirements",
}


def validate(manifest: Json) -> Json:
    if set(manifest) != FIELDS or (
        manifest.get("contract") != 3
        or manifest.get("kind") != "frozen-profile"
        or manifest.get("registration") != "prospective"
    ):
        raise ValueError("Expected prospective frozen-profile contract 3")
    for key, length in (("revision", 40), ("weights_sha256", 64), ("config_sha256", 64)):
        if not isinstance(manifest[key], str) or not re.fullmatch(
            "[0-9a-f]{" + str(length) + "}", manifest[key]
        ):
            raise ValueError(f"Invalid {key}")
    if manifest["case"] not in ("heat_neumann_homogeneous2d", "heat_neumann_polynomial2d"):
        raise ValueError("Frozen profile supports only the registered 2D controls")
    for key in ("trial", "weights"):
        if not isinstance(manifest[key], str):
            raise ValueError("Expected relative source path")
        inside(Path.cwd(), manifest[key])
    if not manifest["trial"].startswith(".gpu-jobs/") or "/" in manifest["weights"]:
        raise ValueError("Expected one checkpoint inside an existing queue trial")
    if type(manifest["points"]) is not int or not 1 <= manifest["points"] <= 4096:
        raise ValueError("Frozen profile point count must be 1..4096")
    if manifest["environment_requirements"] != {
        "torch": "2.1.0+cu118",
        "cuda_runtime": "11.8",
        "default_dtype": "torch.float32",
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }:
        raise ValueError("Frozen profile requires the existing FP32/TF32-off stack")
    return manifest


def plan(manifest: Json) -> Json:
    validate(manifest)
    return dict(
        contract=1,
        kind="frozen-profile-plan",
        revision=manifest["revision"],
        cases=1,
        processes=2,
        rules=["autograd", "analytic"],
        points=manifest["points"],
        optimizer_updates=0,
        timing_is_profiled=True,
        numerical_accuracy_claim=False,
    )


def trace_summary(database: Path) -> Json:
    # Exported SQLite is diagnostic data. Read-only; never interpolate external table names.
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        result: Json = {}
        for label, table in (
            ("kernels", "CUPTI_ACTIVITY_KIND_KERNEL"),
            ("copies", "CUPTI_ACTIVITY_KIND_MEMCPY"),
            ("memsets", "CUPTI_ACTIVITY_KIND_MEMSET"),
        ):
            if table not in tables:
                result[label] = {"count": 0, "summed_nanoseconds": 0}
                continue
            count, duration = connection.execute(
                f'SELECT COUNT(*), COALESCE(SUM(end-start),0) FROM "{table}"'
            ).fetchone()
            if duration < 0:
                raise ValueError("Negative CUDA event duration")
            result[label] = {"count": count, "summed_nanoseconds": duration}
    return result


def collect(root: Path, manifest: Json) -> Json:
    validate(manifest)
    report = read_object(root / "profile.json")
    if report.get("manifest") != manifest:
        raise ValueError("Profile manifest differs")
    for trial in report["trials"]:
        for relative, expected in trial["artifact_sha256"].items():
            if digest(inside(root, relative)) != expected:
                raise ValueError("Profile artifact checksum differs")
    compare(report)
    return report


def compare(evidence: Json) -> Json:
    if evidence.get("contract") != 1 or evidence.get("kind") != "frozen-profile-evidence":
        raise ValueError("Expected frozen profile evidence")
    manifest = validate(object_value(evidence["manifest"], "manifest"))
    if evidence.get("environment") != manifest["environment_requirements"]:
        raise ValueError("Profile environment differs")
    trials = evidence.get("trials")
    if not isinstance(trials, list) or [t.get("rule") for t in trials] != ["autograd", "analytic"]:
        raise ValueError("Expected exactly two frozen profile rules")
    for trial in trials:
        if (
            trial.get("weights_sha256") != manifest["weights_sha256"]
            or trial.get("source_unchanged") is not True
        ):
            raise ValueError("Frozen source changed")
        if not isinstance(trial.get("trace"), dict):
            raise ValueError("Missing trace diagnostics")
        if type(trial.get("profile_exit_code")) is not int:
            raise ValueError("Invalid profile exit code")
        workload = trial.get("workload")
        if workload is not None and (
            workload.get("weights_unchanged") is not True
            or workload.get("weights_sha256") != manifest["weights_sha256"]
            or workload.get("revision") != manifest["revision"]
            or workload.get("feature_derivatives") != trial["rule"]
            or workload.get("timing_includes_external_profiler") is not True
        ):
            raise ValueError("Workload provenance differs")
        count = trial["trace"].get("kernels", {}).get("count", 0)
        if type(count) is not int or count < 0:
            raise ValueError("Invalid kernel count")
    passed = all(
        t["profile_exit_code"] == 0
        and t.get("workload") is not None
        and t["trace"].get("kernels", {}).get("count", 0) > 0
        for t in trials
    )
    return dict(
        contract=1,
        kind="frozen-profile-comparison",
        comparison_passed=passed,
        cuda_trace_captured=passed,
        numerical_accuracy_claim=False,
        performance_claim=False,
        rules=2,
    )


def run(root: Path, checkout: Path, output: Path, manifest: Json) -> None:
    validate(manifest)
    if Path.cwd() != root or str(root) != r"C:\Projects\pinn-runner":
        raise ValueError("Frozen profiling requires the guarded Windows root")
    trial = inside(root, manifest["trial"])
    weights = inside(trial, manifest["weights"])
    config = trial / "config.yaml"
    if (
        digest(weights) != manifest["weights_sha256"]
        or digest(config) != manifest["config_sha256"]
    ):
        raise ValueError("Frozen source differs from protocol")
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    actual = dict(
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        default_dtype=str(torch.get_default_dtype()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
    )
    if actual != manifest["environment_requirements"]:
        raise ValueError("Frozen profile runtime differs")
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    for name in ("APPDATA", "LOCALAPPDATA"):
        target = output / name.lower()
        target.mkdir()
        environment[name] = str(target)
    reports = []
    for rule in ("autograd", "analytic"):
        prefix = output / rule
        command = [
            NSYS,
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--stop-on-exit=true",
            "--force-overwrite=false",
            "--output=" + str(prefix),
            str(root / ".venv/Scripts/python.exe"),
            str(checkout / "scripts/profile_convergence_trial.py"),
            "--trial",
            str(trial),
            "--case",
            manifest["case"],
            "--weights",
            manifest["weights"],
            "--output",
            str(output / (rule + "-work")),
            "--points",
            str(manifest["points"]),
            "--feature-derivatives",
            rule,
            "--native-trace",
        ]
        with (output / (rule + ".log")).open("wb") as log:
            process = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        trace: Json = {}
        artifacts = {rule + ".log": digest(output / (rule + ".log"))}
        native = prefix.with_suffix(".nsys-rep")
        if process.returncode == 0 and native.is_file():
            database = prefix.with_suffix(".sqlite")
            with (output / (rule + "-export.log")).open("wb") as log:
                export = subprocess.run(
                    [NSYS, "export", "--type=sqlite", "--output=" + str(database), str(native)],
                    cwd=root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            artifacts[rule + "-export.log"] = digest(output / (rule + "-export.log"))
            if export.returncode == 0 and database.is_file():
                trace = trace_summary(database)
                artifacts[database.name] = digest(database)
            artifacts[native.name] = digest(native)
        work = output / (rule + "-work/report.json")
        frozen: dict[str, Any] | None = read_object(work) if work.is_file() else None
        if frozen is not None:
            if (
                frozen["weights_unchanged"] is not True
                or frozen["weights_sha256"] != manifest["weights_sha256"]
            ):
                raise ValueError("Workload changed weights")
            artifacts[str(work.relative_to(output)).replace("\\", "/")] = digest(work)
        diagnostic_tail = (output / (rule + ".log")).read_text(encoding="utf-8", errors="replace")[
            -1800:
        ]
        reports.append(
            dict(
                diagnostic_tail=diagnostic_tail,
                rule=rule,
                profile_exit_code=process.returncode,
                trace=trace,
                weights_sha256=digest(weights),
                source_unchanged=digest(weights) == manifest["weights_sha256"]
                and digest(config) == manifest["config_sha256"],
                workload=frozen,
                artifact_sha256=artifacts,
            )
        )
    evidence = dict(
        contract=1,
        kind="frozen-profile-evidence",
        manifest=manifest,
        environment=actual,
        trials=reports,
    )
    (output / "profile.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
