"""Fresh public prediction processes, including interpreter and MAT output."""

from __future__ import annotations

import statistics
import subprocess
from pathlib import Path

from .frozen_profile import FIELDS
from .frozen_profile import validate as validate_source
from .schema import Json, digest, inside, number, read_object

EXTRA = {"repeats", "batch_sizes", "space_points", "time_points", "comparison"}
GATES = dict(median_ratio_max=1.10, worst_ratio_max=1.25, peak_ratio_max=1.10)


def validate(m: Json) -> Json:
    if set(m) != FIELDS | EXTRA or m.get("contract") != 8 or m.get("kind") != "public-prediction":
        raise ValueError("Expected public prediction contract8")
    source = {k: v for k, v in m.items() if k not in EXTRA}
    validate_source(
        source
        | dict(contract=3, kind="frozen-profile", case="heat_neumann_polynomial2d", points=1024)
    )
    if (
        any(type(m[k]) is not int for k in ("points", "repeats", "space_points", "time_points"))
        or m["case"] != "heat_neumann_polynomial3d"
        or m["points"] != 152303
        or m["repeats"] != 3
        or m["batch_sizes"] != [128, 2048]
        or m["space_points"] != 17
        or m["time_points"] != 31
        or m["comparison"] != GATES
    ):
        raise ValueError("Only the registered 3D cold prediction matrix is supported")
    return m


def plan(m: Json) -> Json:
    validate(m)
    return dict(
        contract=1,
        kind="public-prediction-plan",
        revision=m["revision"],
        processes=12,
        optimizer_updates=0,
        points=152303,
        batch_sizes=[128, 2048],
        last_batch_sizes=[111, 751],
        includes_interpreter_and_mat=True,
        os_caches_flushed=False,
    )


def inventory() -> list[tuple[int, int, str]]:
    return [
        (repeat, batch, rule)
        for repeat in range(3)
        for batch in (128, 2048)
        for rule in (
            ("eager", "neumann_analytic") if repeat % 2 == 0 else ("neumann_analytic", "eager")
        )
    ]


def compare(e: Json) -> Json:
    m = validate(e["manifest"])
    if e.get("contract") != 1 or e.get("kind") != "public-prediction-evidence":
        raise ValueError("Expected public prediction evidence")
    if (
        e["revision"] != m["revision"]
        or e["environment"] != m["environment_requirements"]
        or e["gpu"] != "NVIDIA GeForce RTX 4070 Ti"
        or e["source_unchanged"] is not True
        or e["converted_state_equal"] is not True
        or e["cross_batch_close"] is not True
    ):
        raise ValueError("Prediction provenance or field control differs")
    trials = e["trials"]
    if [(t["repeat"], t["batch"], t["rule"]) for t in trials] != inventory():
        raise ValueError("Incomplete fresh prediction matrix")
    pairs = []
    for batch in (128, 2048):
        ratios = []
        allocated_ratios = []
        reserved_ratios = []
        for repeat in range(3):
            pair = {t["rule"]: t for t in trials if t["batch"] == batch and t["repeat"] == repeat}
            a, b = pair["eager"], pair["neumann_analytic"]
            for trial in (a, b):
                if (
                    trial["environment"] != m["environment_requirements"]
                    or trial["gpu"] != e["gpu"]
                    or trial["exit_code"] != 0
                    or trial["field_shape"] != [4913, 31]
                    or trial["finite"] is not True
                ):
                    raise ValueError("Public prediction failed")
                number(trial["seconds"], "public seconds", positive=True)
                number(trial["peak_allocated_bytes"], "public peak", positive=True)
                if (
                    number(trial["peak_reserved_bytes"], "public reserved", positive=True)
                    < trial["peak_allocated_bytes"]
                ):
                    raise ValueError("Public peak reserved is below allocated")
            if a["field_sha256"] != b["field_sha256"]:
                raise ValueError("Paired public fields differ")
            ratios.append(b["seconds"] / a["seconds"])
            allocated_ratios.append(b["peak_allocated_bytes"] / a["peak_allocated_bytes"])
            reserved_ratios.append(b["peak_reserved_bytes"] / a["peak_reserved_bytes"])
        pairs.append(
            dict(
                batch=batch,
                median_ratio=statistics.median(ratios),
                worst_ratio=max(ratios),
                worst_allocated_ratio=max(allocated_ratios),
                worst_reserved_ratio=max(reserved_ratios),
            )
        )
    passed = all(
        p["median_ratio"] <= GATES["median_ratio_max"]
        and p["worst_ratio"] <= GATES["worst_ratio_max"]
        and p["worst_allocated_ratio"] <= GATES["peak_ratio_max"]
        and p["worst_reserved_ratio"] <= GATES["peak_ratio_max"]
        for p in pairs
    )
    return dict(
        contract=1,
        kind="public-prediction-comparison",
        trials=12,
        pairs=pairs,
        comparison_passed=passed,
        training_speedup_claim=False,
        scope="Fresh processes, warm OS/driver caches",
    )


def collect(root: Path, m: Json) -> Json:
    validate(m)
    e = read_object(root / "public-prediction.json")
    if e["manifest"] != m:
        raise ValueError("Public prediction manifest differs")
    for relative, sha in e["artifact_sha256"].items():
        if digest(inside(root, relative)) != sha:
            raise ValueError("Public prediction artifact changed")
    compare(e)
    return e


def run(root: Path, checkout: Path, output: Path, m: Json) -> None:
    validate(m)
    if Path.cwd() != root or str(root) != r"C:\Projects\pinn-runner":
        raise ValueError("Guarded Windows root required")
    subprocess.run(
        [
            str(root / ".venv/Scripts/python.exe"),
            "-B",
            str(checkout / "scripts/measure_public_prediction.py"),
            "--manifest",
            str(output.parent / "manifest.json"),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
    )
