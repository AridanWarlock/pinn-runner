"""Registered unprofiled timing of existing weights; no optimization updates."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

from .frozen_profile import validate as validate_profile
from .schema import Json, digest, inside, number, object_value, read_object

FIELDS = {
    "contract",
    "kind",
    "registration",
    "revision",
    "sources",
    "repeats",
    "points",
    "prediction_points",
    "blocks",
    "block_seconds",
    "prediction_block_seconds",
    "environment_requirements",
    "comparison",
}
SOURCE_FIELDS = {"case", "seed", "trial", "weights", "weights_sha256", "config_sha256"}
CASES = {f"heat_neumann_{kind}{d}d" for kind in ("homogeneous", "polynomial") for d in (2, 3)}


def validate(manifest: Json) -> Json:
    if (
        set(manifest) != FIELDS
        or manifest.get("contract") != 4
        or manifest.get("kind") != "frozen-timing"
        or manifest.get("registration") != "prospective"
    ):
        raise ValueError("Expected prospective frozen-timing contract 4")
    sources = manifest["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= 8:
        raise ValueError("Expected 1..8 frozen sources")
    identities = set()
    for raw in sources:
        source = object_value(raw, "source")
        if (
            set(source) != SOURCE_FIELDS
            or source["case"] not in CASES
            or type(source["seed"]) is not int
            or source["seed"] < 0
        ):
            raise ValueError("Invalid frozen source")
        source_identity = (source["case"], source["seed"])
        if source_identity in identities:
            raise ValueError("Duplicate frozen source")
        identities.add(source_identity)
        # Reuse strict path/hash/precision validation, without extending profile case support.
        validate_profile(
            dict(
                contract=3,
                kind="frozen-profile",
                registration="prospective",
                revision=manifest["revision"],
                case="heat_neumann_polynomial2d",
                points=manifest["points"],
                environment_requirements=manifest["environment_requirements"],
                **{k: source[k] for k in ("trial", "weights", "weights_sha256", "config_sha256")},
            )
        )
    for key, expected in (
        ("repeats", 3),
        ("points", 1024),
        ("prediction_points", 8193),
        ("blocks", 9),
    ):
        if type(manifest[key]) is not int or manifest[key] != expected:
            raise ValueError(f"Frozen timing requires {key}={expected}")
    if manifest["block_seconds"] != 6.0 or manifest["prediction_block_seconds"] != 0.75:
        raise ValueError("Frozen timing durations must be 6 and 0.75 seconds")
    if manifest["comparison"] != dict(
        median_objective_ratio_max=0.95,
        worst_objective_ratio_max=1.10,
        median_prediction_ratio_max=1.10,
        worst_allocated_ratio_max=1.10,
    ):
        raise ValueError("Frozen timing gates differ from supported prospective protocol")
    return manifest


def expected(manifest: Json) -> list[tuple[int, int, str]]:
    return [
        (i, r, rule)
        for i in range(len(manifest["sources"]))
        for r in range(manifest["repeats"])
        for rule in (("autograd", "analytic") if (i + r) % 2 == 0 else ("analytic", "autograd"))
    ]


def plan(manifest: Json) -> Json:
    validate(manifest)
    count = len(expected(manifest))
    return dict(
        contract=1,
        kind="frozen-timing-plan",
        revision=manifest["revision"],
        sources=len(manifest["sources"]),
        processes=count,
        optimizer_updates=0,
        minimum_measured_seconds=count * 9 * (6 + 2 * 0.75),
        batch_sizes=[128, 2048],
        last_batch_points=1,
        performance_requires_collection=True,
    )


def compare(evidence: Json) -> Json:
    manifest = validate(object_value(evidence["manifest"], "manifest"))
    trials = evidence["trials"]
    if evidence.get("kind") != "frozen-timing-evidence" or evidence.get("contract") != 1:
        raise ValueError("Expected frozen timing evidence")
    if [(t["source_index"], t["repeat"], t["rule"]) for t in trials] != expected(manifest):
        raise ValueError("Incomplete frozen timing matrix")
    for trial in trials:
        source = manifest["sources"][trial["source_index"]]
        if (
            trial["environment"] != manifest["environment_requirements"]
            or trial["revision"] != manifest["revision"]
            or trial["source"] != source
        ):
            raise ValueError("Frozen timing provenance differs")
        if trial["weights_unchanged"] is not True or trial["source_unchanged"] is not True:
            raise ValueError("Frozen timing source changed")
        if trial["profiled"] is not False or trial["independent_autograd_control"] is None:
            raise ValueError("Missing unprofiled independent control")
        control = object_value(trial["independent_autograd_control"], "control")
        relative = number(control["gradient_relative_l2_difference"], "gradient difference")
        maximum = number(control["gradient_max_absolute_difference"], "gradient difference")
        number(control["loss_absolute_difference"], "loss difference")
        if relative > 2e-4 and maximum > 2e-7:
            raise ValueError("Frozen independent gradient control failed")
        number(trial["cold_ready_seconds"], "cold ready", positive=True)
        number(trial["cold_first_prediction_seconds"], "cold prediction", positive=True)
        groups = [trial["objective"]] + [trial["prediction"][str(batch)] for batch in (128, 2048)]
        for j, group in enumerate(groups):
            number(group["peak_allocated_bytes"], "allocated memory", positive=True)
            number(group["peak_reserved_bytes"], "reserved memory", positive=True)
            samples = group["samples"]
            budget = manifest["block_seconds"] if j == 0 else manifest["prediction_block_seconds"]
            if len(samples) != 9 or any(
                type(s["calls"]) is not int
                or s["calls"] <= 0
                or not isinstance(s["seconds"], (int, float))
                or not budget <= s["seconds"] < budget + 60
                for s in samples
            ):
                raise ValueError("Invalid timing samples")
            measured = statistics.median(s["seconds"] / s["calls"] for s in samples)
            if group["median_seconds"] != measured:
                raise ValueError("Cached timing median differs")
    pairs = []
    for index, source in enumerate(manifest["sources"]):
        for repeat in range(3):
            arms = {
                t["rule"]: t
                for t in trials
                if t["source_index"] == index and t["repeat"] == repeat
            }
            a, b = arms["autograd"], arms["analytic"]
            if (
                a["state_sha256"] != b["state_sha256"]
                or a["prediction_sha256"] != b["prediction_sha256"]
                or a["points_sha256"] != b["points_sha256"]
            ):
                raise ValueError("Frozen pair states or points differ")
            pairs.append(
                dict(
                    case=source["case"],
                    seed=source["seed"],
                    repeat=repeat,
                    objective_ratio=b["objective"]["median_seconds"]
                    / a["objective"]["median_seconds"],
                    prediction128_ratio=b["prediction"]["128"]["median_seconds"]
                    / a["prediction"]["128"]["median_seconds"],
                    prediction2048_ratio=b["prediction"]["2048"]["median_seconds"]
                    / a["prediction"]["2048"]["median_seconds"],
                    allocated_ratio=b["objective"]["peak_allocated_bytes"]
                    / a["objective"]["peak_allocated_bytes"],
                )
            )
    cases = []
    for name in sorted({p["case"] for p in pairs}):
        rows = [p for p in pairs if p["case"] == name]
        row = dict(
            case=name,
            pairs=len(rows),
            median_objective_ratio=statistics.median(p["objective_ratio"] for p in rows),
            worst_objective_ratio=max(p["objective_ratio"] for p in rows),
            median_prediction128_ratio=statistics.median(p["prediction128_ratio"] for p in rows),
            median_prediction2048_ratio=statistics.median(p["prediction2048_ratio"] for p in rows),
            worst_allocated_ratio=max(p["allocated_ratio"] for p in rows),
        )
        row["passed"] = (
            row["median_objective_ratio"] <= 0.95
            and row["worst_objective_ratio"] <= 1.10
            and max(row["median_prediction128_ratio"], row["median_prediction2048_ratio"]) <= 1.10
            and row["worst_allocated_ratio"] <= 1.10
        )
        cases.append(row)
    return dict(
        contract=1,
        kind="frozen-timing-comparison",
        comparison_passed=all(c["passed"] for c in cases),
        numerical_accuracy_claim=False,
        cases=cases,
        details=pairs,
    )


def collect(root: Path, manifest: Json) -> Json:
    validate(manifest)
    evidence = read_object(root / "timing.json")
    if evidence["manifest"] != manifest:
        raise ValueError("Timing manifest differs")
    for relative, checksum in evidence["artifact_sha256"].items():
        if digest(inside(root, relative)) != checksum:
            raise ValueError("Timing artifact checksum differs")
    compare(evidence)
    return evidence


def run(root: Path, checkout: Path, output: Path, manifest: Json) -> None:
    validate(manifest)
    if Path.cwd() != root or str(root) != r"C:\Projects\pinn-runner":
        raise ValueError("Timing requires guarded Windows root")
    for source in manifest["sources"]:
        source_directory = inside(root, source["trial"])
        if (
            digest(inside(source_directory, source["weights"])) != source["weights_sha256"]
            or digest(source_directory / "config.yaml") != source["config_sha256"]
        ):
            raise ValueError("Frozen source differs before timing")
    output.mkdir(parents=True, exist_ok=False)
    trials = []
    artifacts = {}
    matrix = expected(manifest)
    for i, repeat, rule in matrix:
        label = f"s{i}-r{repeat}-{rule}"
        destination = output / (label + ".json")
        started = time.perf_counter()
        with (output / (label + ".log")).open("wb") as log:
            subprocess.run(
                [
                    str(root / ".venv/Scripts/python.exe"),
                    "-B",
                    str(checkout / "scripts/benchmark_frozen_timing.py"),
                    "--manifest",
                    str(output.parent / "manifest.json"),
                    "--source-index",
                    str(i),
                    "--repeat",
                    str(repeat),
                    "--rule",
                    rule,
                    "--output",
                    str(destination),
                ],
                cwd=root,
                env=dict(os.environ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        trial = read_object(destination)
        trial["process_seconds"] = time.perf_counter() - started
        trials.append(trial)
        artifacts[destination.name] = digest(destination)
        artifacts[label + ".log"] = digest(output / (label + ".log"))
        progress = dict(
            completed_trials=len(trials),
            expected_trials=len(matrix),
            case=manifest["sources"][i]["case"],
            repeat=repeat,
            rule=rule,
        )
        (output.parent / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
        print(json.dumps(progress), flush=True)
    evidence = dict(
        contract=1,
        kind="frozen-timing-evidence",
        manifest=manifest,
        trials=trials,
        artifact_sha256=artifacts,
    )
    compare(evidence)
    (output / "timing.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
