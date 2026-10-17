"""Validate explicit evidence contracts at the JSON boundary."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

Json = dict[str, Any]
ARM_FIELDS = ("backend", "lifting", "backward", "background", "spatial_encoding")
IDENTITY_FIELDS = ("case", "seed", *ARM_FIELDS)


def object_value(value: Any, label: str) -> Json:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError(f"{label}: expected object")
    return value


def read_object(path: Path) -> Json:
    return object_value(json.loads(path.read_text(encoding="utf-8")), str(path))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(
            f"{label}: expected {'positive' if positive else 'nonnegative'} finite number"
        )
    return result


def identity(trial: Json) -> tuple[str | int, ...]:
    values: list[str | int] = []
    for key in IDENTITY_FIELDS:
        value = trial.get(key)
        if key == "seed":
            if type(value) is not int or value < 0:
                raise ValueError("seed: expected nonnegative integer")
        elif not isinstance(value, str) or not value:
            raise ValueError(f"{key}: expected nonempty string")
        values.append(value)
    return tuple(values)


def inside(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative or ":" in relative:
        raise ValueError("Use a relative POSIX path inside the study")
    path = (root / relative).resolve()
    if (
        Path(relative).is_absolute()
        or not path.is_relative_to(root.resolve())
        or path == root.resolve()
    ):
        raise ValueError("Path escapes the study directory")
    return path


def load_manifest(path: Path) -> Json:
    return validate_manifest(read_object(path), path.parent)


def validate_manifest(manifest: Json, base: Path) -> Json:
    required = {
        "contract",
        "registration",
        "revision",
        "reports",
        "expected_trials",
        "report_parameters",
        "case_limits",
        "physical_limits",
        "comparison",
    }
    if set(manifest) != required or manifest["contract"] != 1:
        raise ValueError("Manifest contract 1 with exactly the documented fields required")
    if manifest["registration"] not in ("prospective", "retrospective_import"):
        raise ValueError("registration must distinguish prospective from retrospective import")
    if not isinstance(manifest["revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", manifest["revision"]
    ):
        raise ValueError("revision: full lowercase Git SHA required")
    reports = manifest["reports"]
    if (
        not isinstance(reports, list)
        or not reports
        or not all(isinstance(p, str) for p in reports)
    ):
        raise ValueError("reports: nonempty list of relative paths required")
    if len(set(reports)) != len(reports):
        raise ValueError("Duplicate report paths")
    for relative in reports:
        inside(base, relative)
    expected = manifest["expected_trials"]
    if not isinstance(expected, list) or not expected:
        raise ValueError("expected_trials: explicit nonempty trial matrix required")
    keys = [identity(object_value(t, "expected trial")) for t in expected]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate expected trial")
    parameters = object_value(manifest["report_parameters"], "report_parameters")
    required_parameters = {
        "adam_steps",
        "lbfgs_snapshots",
        "constraints",
        "points_override",
        "lbfgs_tolerance_change",
        "execution_lifecycle",
    }
    if set(parameters) != required_parameters:
        raise ValueError(
            "report_parameters must fix budget, points, constraints, lifecycle and tolerance"
        )
    if type(parameters["adam_steps"]) is not int or parameters["adam_steps"] <= 0:
        raise ValueError("adam_steps: positive integer required")
    steps = parameters["lbfgs_snapshots"]
    if not isinstance(steps, list) or not steps or any(type(s) is not int or s < 0 for s in steps):
        raise ValueError("lbfgs_snapshots: increasing nonnegative integers required")
    if steps != sorted(set(steps)):
        raise ValueError("lbfgs_snapshots: increasing unique integers required")
    points = parameters["points_override"]
    if points is not None and (type(points) is not int or points <= 0):
        raise ValueError("points_override: positive integer or null required")
    number(parameters["lbfgs_tolerance_change"], "tolerance_change", positive=True)
    if parameters["execution_lifecycle"] != "isolated_trials":
        raise ValueError("Only isolated trials are accepted for paired evidence")
    if not isinstance(parameters["constraints"], str) or not parameters["constraints"]:
        raise ValueError("constraints: explicit name required")
    case_limits = object_value(manifest["case_limits"], "case_limits")
    if set(case_limits) != {t["case"] for t in expected}:
        raise ValueError("case_limits must cover exactly the expected cases")
    for limits in case_limits.values():
        values = object_value(limits, "case limits")
        if not {"relative_l2", "final_relative_l2", "pde_rmse"} <= values.keys():
            raise ValueError("Global/final/PDE limits required")
        for key, value in values.items():
            number(value, key, positive=True)
    physical = object_value(manifest["physical_limits"], "physical_limits")
    if set(physical) != {"initial_rmse", "boundary_max"}:
        raise ValueError("initial_rmse and boundary_max physical limits required")
    for key, value in physical.items():
        number(value, key, positive=True)
    comparison = object_value(manifest["comparison"], "comparison")
    if set(comparison) != {
        "baseline",
        "candidate",
        "quality_factor",
        "median_training_ratio_max",
        "worst_training_ratio_max",
    }:
        raise ValueError("Explicit paired comparison and timing gates required")
    for name in ("baseline", "candidate"):
        arm = object_value(comparison[name], name)
        if set(arm) != set(ARM_FIELDS) or not all(isinstance(v, str) and v for v in arm.values()):
            raise ValueError("Comparison arms must specify all arm fields")
    if comparison["baseline"] == comparison["candidate"]:
        raise ValueError("Baseline and candidate must differ")
    if any(
        {k: t[k] for k in ARM_FIELDS} not in (comparison["baseline"], comparison["candidate"])
        for t in expected
    ):
        raise ValueError("Every expected trial must belong to a declared comparison arm")
    for case, seed in {(t["case"], t["seed"]) for t in expected}:
        if sum(t["case"] == case and t["seed"] == seed for t in expected) != 2:
            raise ValueError("Every case/seed must have exactly one baseline and one candidate")
    for key in ("quality_factor", "median_training_ratio_max", "worst_training_ratio_max"):
        number(comparison[key], key, positive=True)
    return manifest
