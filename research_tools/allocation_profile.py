"""A bounded diagnostic pair using Torch allocator events, not CUDA kernel timing."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .frozen_profile import validate as validate_source
from .schema import Json, digest, inside, number, read_object

STAGES = (
    "network_and_features",
    "background_reconstruction",
    "physical_residual",
    "loss_reduction",
    "parameter_gradient",
)


def validate(manifest: Json) -> Json:
    if manifest.get("contract") != 7 or manifest.get("kind") != "allocation-profile":
        raise ValueError("Expected allocation profile contract7")
    validate_source(manifest | dict(contract=3, kind="frozen-profile"))
    if manifest["case"] != "heat_neumann_polynomial2d" or manifest["points"] != 1024:
        raise ValueError("Only the saved polynomial2D/1024 diagnostic is registered")
    return manifest


def plan(manifest: Json) -> Json:
    validate(manifest)
    return dict(
        contract=1,
        kind="allocation-profile-plan",
        revision=manifest["revision"],
        processes=2,
        profiled_operations=2,
        total_frozen_objective_calls=8,
        unprofiled_controls=6,
        optimizer_updates=0,
        nsight=False,
        performance_claim=False,
    )


def compare(evidence: Json) -> Json:
    manifest = validate(evidence["manifest"])
    if evidence.get("contract") != 1 or evidence.get("kind") != "allocation-profile-evidence":
        raise ValueError("Expected allocation evidence")
    if evidence["environment"] != manifest["environment_requirements"]:
        raise ValueError("Allocation environment differs")
    trials = evidence["trials"]
    if [t["rule"] for t in trials] != ["autograd", "analytic"]:
        raise ValueError("Incomplete allocation pair")
    fingerprints = []
    for trial in trials:
        if (
            trial["gpu"] != "NVIDIA GeForce RTX 4070 Ti"
            or trial["environment"] != manifest["environment_requirements"]
            or trial["revision"] != manifest["revision"]
            or trial["weights_sha256"] != manifest["weights_sha256"]
            or trial["source_unchanged"] is not True
            or trial["decomposed_gradient_passed"] is not True
        ):
            raise ValueError("Allocation provenance or independent gradient check differs")
        control = trial["independent_autograd_control"]
        if number(control["loss_absolute_difference"], "loss difference") > 1e-8 or (
            number(control["gradient_relative_l2_difference"], "gradient relative") > 2e-4
            and number(control["gradient_max_absolute_difference"], "gradient absolute") > 2e-7
        ):
            raise ValueError("Independent autograd control failed")
        fingerprints.append(trial["state_sha256"])
        profile = trial["profiling"]
        if (
            profile["profiled"] is not True
            or profile["performance_claim"] is not False
            or profile["cuda_kernel_timing"] is not False
        ):
            raise ValueError("Allocator diagnostics are not kernel timing")
        if [r["stage"] for r in profile["stages"]] != list(STAGES):
            raise ValueError("Missing allocation stage")
        for row in profile["stages"]:
            for key, value in row.items():
                if key != "stage" and (type(value) is not int or value < 0):
                    raise ValueError("Invalid allocator counter")
            if row["peak_bytes"] < max(row["before_bytes"], row["after_bytes"]):
                raise ValueError("Invalid stage allocation peak")
        if type(profile["cuda_memory_events"]) is not int or profile["cuda_memory_events"] < 0:
            raise ValueError("Invalid CUDA memory count")
    if len(set(fingerprints)) != 1:
        raise ValueError("Allocation pair uses different model states")
    for trial in trials:
        if trial["profiling"]["cuda_memory_events"] != sum(
            r["allocation_events"] + r["free_events"] for r in trial["profiling"]["stages"]
        ):
            raise ValueError("Allocator events disagree with stage counters")
    captured = all(r["allocation_events"] > 0 for t in trials for r in t["profiling"]["stages"])
    return dict(
        contract=1,
        kind="allocation-profile-comparison",
        trials=2,
        comparison_passed=captured,
        cuda_allocator_events_captured=captured,
        performance_claim=False,
    )


def collect(root: Path, manifest: Json) -> Json:
    validate(manifest)
    evidence = read_object(root / "allocation.json")
    if evidence["manifest"] != manifest:
        raise ValueError("Allocation manifest differs")
    for relative, sha in evidence["artifact_sha256"].items():
        if digest(inside(root, relative)) != sha:
            raise ValueError("Allocation artifact changed")
    compare(evidence)
    return evidence


def run(root: Path, checkout: Path, output: Path, manifest: Json) -> None:
    validate(manifest)
    if Path.cwd() != root or str(root) != r"C:\Projects\pinn-runner":
        raise ValueError("Guarded Windows root required")
    trial = inside(root, manifest["trial"])
    weights = inside(trial, manifest["weights"])
    config = trial / "config.yaml"
    if (
        digest(weights) != manifest["weights_sha256"]
        or digest(config) != manifest["config_sha256"]
    ):
        raise ValueError("Saved allocation source differs")
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
        raise ValueError("Allocation runtime differs")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    trials = []
    env = dict(os.environ)
    for name in ("APPDATA", "LOCALAPPDATA"):
        target = output / name.lower()
        target.mkdir()
        env[name] = str(target)
    for rule in ("autograd", "analytic"):
        destination = output / rule
        command = [
            str(root / ".venv/Scripts/python.exe"),
            "-B",
            str(checkout / "scripts/profile_allocations.py"),
            "--trial",
            str(trial),
            "--weights",
            manifest["weights"],
            "--rule",
            rule,
            "--output",
            str(destination),
        ]
        with (output / (rule + ".log")).open("wb") as log:
            subprocess.run(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        report = read_object(destination / "report.json")
        artifacts[rule + "/report.json"] = digest(destination / "report.json")
        artifacts[rule + ".log"] = digest(output / (rule + ".log"))
        for relative, sha in report["artifact_sha256"].items():
            if digest(inside(destination, relative)) != sha:
                raise ValueError("Allocator trace changed")
            artifacts[rule + "/" + relative] = sha
        trials.append(report)
    if (
        digest(weights) != manifest["weights_sha256"]
        or digest(config) != manifest["config_sha256"]
    ):
        raise ValueError("Allocation source changed after profiling")
    evidence = dict(
        contract=1,
        kind="allocation-profile-evidence",
        manifest=manifest,
        trials=trials,
        environment=actual,
        artifact_sha256=artifacts,
    )
    compare(evidence)
    (output / "allocation.json").write_text(json.dumps(evidence, indent=2) + "\n")
