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
    if "coefficient_representation" in trial:
        representation = trial["coefficient_representation"]
        if representation not in ("formula", "table"):
            raise ValueError("Invalid coefficient representation")
        values.append(representation)
    if "points_override" in trial:
        points = trial["points_override"]
        if type(points) is not int or points <= 0:
            raise ValueError("Trial points_override must be positive")
        values.append(points)
    return tuple(values)


def arm_fields(manifest: Json) -> tuple[str, ...]:
    if "physics" in manifest:
        return (*ARM_FIELDS, "coefficient_representation")
    return (*ARM_FIELDS, "points_override") if "density" in manifest else ARM_FIELDS


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
    if manifest.get("contract") == 8:
        from .public_prediction import validate as public_validate

        return public_validate(manifest)
    if manifest.get("contract") == 7:
        from .allocation_profile import validate as allocation_validate

        return allocation_validate(manifest)
    if manifest.get("contract") == 6:
        from .numerical_acceptance import validate

        return validate(manifest)
    if manifest.get("contract") == 5:
        from .lbfgs_timing import validate

        return validate(manifest)
    if manifest.get("contract") == 4:
        from .frozen_timing import validate

        return validate(manifest)
    if manifest.get("contract") == 3:
        from .frozen_profile import validate

        return validate(manifest)
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
    from benchmarks.catalog import UNFORCED_EVALUATION, UNFORCED_VISCOSITIES

    absolute = "absolute_quality" in manifest
    if absolute:
        required = (required - {"comparison"}) | {"absolute_quality", "problem_sha256"}
    if "layer_evaluation" in manifest:
        required.add("layer_evaluation")
    if "hypothesis_evaluation" in manifest:
        required.add("hypothesis_evaluation")
    if manifest.get("contract") == 2:
        required |= {"case_faces", "environment_requirements", "network"}
        if "density" in manifest:
            required.add("density")
        if "physics" in manifest:
            required.add("physics")
    if set(manifest) != required or manifest["contract"] not in (1, 2):
        raise ValueError("Manifest contract 1/2 with exactly the documented fields required")
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
    from benchmarks.catalog import SURROGATE_EVALUATIONS

    surrogate = bool(set(case_limits) & SURROGATE_EVALUATIONS.keys())
    if surrogate or "hypothesis_evaluation" in manifest:
        evaluation = object_value(manifest.get("hypothesis_evaluation"), "hypothesis_evaluation")
        if (
            not absolute
            or len(case_limits) != 1
            or next(iter(case_limits)) not in SURROGATE_EVALUATIONS
            or json.dumps(
                {k: v for k, v in evaluation.items() if k != "reference_sha256"}, sort_keys=True
            )
            != json.dumps(SURROGATE_EVALUATIONS[next(iter(case_limits))], sort_keys=True)
            or not isinstance(evaluation.get("reference_sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", evaluation["reference_sha256"])
        ):
            raise ValueError("Registered per-hypothesis evaluation and reference hash required")
    unforced = bool(set(case_limits) & UNFORCED_VISCOSITIES.keys())
    if unforced or "layer_evaluation" in manifest:
        evaluation = object_value(manifest.get("layer_evaluation"), "layer_evaluation")
        if (
            not absolute
            or not set(case_limits) <= UNFORCED_VISCOSITIES.keys()
            or json.dumps(
                {k: v for k, v in evaluation.items() if k != "reference_sha256"}, sort_keys=True
            )
            != json.dumps(UNFORCED_EVALUATION, sort_keys=True)
            or not isinstance(evaluation.get("reference_sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", evaluation["reference_sha256"])
        ):
            raise ValueError("Registered unforced layer evaluation and reference hash required")
    physical = object_value(manifest["physical_limits"], "physical_limits")
    if set(physical) != {"initial_rmse", "boundary_max"}:
        raise ValueError("initial_rmse and boundary_max physical limits required")
    for key, value in physical.items():
        number(value, key, positive=True)
    if absolute:
        if manifest["contract"] != 2 or "density" in manifest or "physics" in manifest:
            raise ValueError("Absolute quality requires contract2 without paired studies")
        arm = object_value(manifest["absolute_quality"], "absolute_quality")
        expected_arm = dict(
            backend="eager",
            lifting="recursive",
            backward="all",
            background="general",
            spatial_encoding="identity",
        )
        hard_burgers = parameters["constraints"] == "hard_initial_dirichlet" and set(
            case_limits
        ) == {"burgers"}
        hard_burgers |= parameters["constraints"] == "hard_initial_dirichlet" and unforced
        if arm != expected_arm or (parameters["constraints"] != "soft" and not hard_burgers):
            raise ValueError(
                "Absolute quality requires soft or exact Burgers Dirichlet with eager execution"
            )
        if any(
            set(t) != {"case", "seed", *ARM_FIELDS} or {k: t[k] for k in ARM_FIELDS} != arm
            for t in expected
        ):
            raise ValueError("Every trial must use the declared single configuration")
        if len({(t["case"], t["seed"]) for t in expected}) != len(expected):
            raise ValueError("One trial per case/seed required")
        sources = object_value(manifest["problem_sha256"], "problem_sha256")
        if set(sources) != set(case_limits) or any(
            not isinstance(v, str) or re.fullmatch("[0-9a-f]{64}", v) is None
            for v in sources.values()
        ):
            raise ValueError("Pin every problem source SHA256")
    else:
        comparison = object_value(manifest["comparison"], "comparison")
        comparison_fields = {"baseline", "candidate", "quality_factor"}
        if "physics" not in manifest:
            comparison_fields |= {"median_training_ratio_max", "worst_training_ratio_max"}
        if set(comparison) != comparison_fields:
            raise ValueError("Explicit paired comparison and timing gates required")
        fields = arm_fields(manifest)
        for name in ("baseline", "candidate"):
            arm = object_value(comparison[name], name)
            if set(arm) != set(fields) or not all(
                isinstance(arm[k], str) and arm[k] for k in ARM_FIELDS
            ):
                raise ValueError("Comparison arms must specify all arm fields")
        if comparison["baseline"] == comparison["candidate"]:
            raise ValueError("Baseline and candidate must differ")
        if any(
            {k: t[k] for k in fields} not in (comparison["baseline"], comparison["candidate"])
            for t in expected
        ):
            raise ValueError("Every expected trial must belong to a declared comparison arm")
        for case, seed in {(t["case"], t["seed"]) for t in expected}:
            if sum(t["case"] == case and t["seed"] == seed for t in expected) != 2:
                raise ValueError(
                    "Every case/seed must have exactly one baseline and one candidate"
                )
        for key in comparison_fields - {"baseline", "candidate"}:
            number(comparison[key], key, positive=True)
    if "physics" in manifest:
        from benchmarks.conductivity import physics

        if manifest["contract"] != 2 or "density" in manifest:
            raise ValueError("Coefficient comparison requires contract2 without density")
        definitions = object_value(manifest["physics"], "physics")
        if set(definitions) != set(case_limits) or any(
            definitions[case] != physics(case) for case in case_limits
        ):
            raise ValueError("Physics differs from registered conductivity cases")
        left, right = comparison["baseline"], comparison["candidate"]
        if (
            left["coefficient_representation"] != "formula"
            or right["coefficient_representation"] != "table"
        ):
            raise ValueError("Coefficient comparison requires formula/table arms")
        if any(left[k] != right[k] for k in ARM_FIELDS):
            raise ValueError("Coefficient comparison must preserve execution and features")
        if any(t.get("coefficient_representation") not in ("formula", "table") for t in expected):
            raise ValueError("Expected coefficient representation required")
    if "density" in manifest:
        density = object_value(manifest["density"], "density")
        if set(density) != {"training_seconds", "snapshot_seconds", "sampling", "prefix_points"}:
            raise ValueError("Density requires training/snapshot seconds and sampling")
        budget = number(density["training_seconds"], "training_seconds", positive=True)
        interval = number(density["snapshot_seconds"], "snapshot_seconds", positive=True)
        if interval >= budget or density["sampling"] != "nested_uniform":
            raise ValueError("Density requires nested_uniform and snapshots within budget")
        left, right = (comparison[k] for k in ("baseline", "candidate"))
        if any(left[k] != right[k] for k in ARM_FIELDS):
            raise ValueError("Density comparison must change only point count")
        if any(type(a["points_override"]) is not int for a in (left, right)) or not (
            0 < left["points_override"] < right["points_override"]
        ):
            raise ValueError("Density requires increasing positive point counts")
        if parameters["points_override"] is not None:
            raise ValueError("Density point counts belong to arms, not shared parameters")
        if (
            type(density["prefix_points"]) is not int
            or density["prefix_points"] != left["points_override"]
        ):
            raise ValueError("Density prefix must equal baseline point count")
    if manifest["contract"] == 2:
        faces = object_value(manifest["case_faces"], "case_faces")
        if set(faces) != set(case_limits):
            raise ValueError("case_faces must cover exactly the cases")
        for case_faces in faces.values():
            if not isinstance(case_faces, list) or not case_faces:
                raise ValueError("Expected boundary faces required")
            ids = [face_identity(object_value(face, "face")) for face in case_faces]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate expected boundary face")
        environment = object_value(
            manifest["environment_requirements"], "environment_requirements"
        )
        if set(environment) != {
            "torch",
            "cuda_runtime",
            "default_dtype",
            "float32_matmul_precision",
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
        }:
            raise ValueError("Pin runtime, dtype and precision flags explicitly")
        if not all(
            isinstance(environment[k], str) and environment[k]
            for k in ("torch", "cuda_runtime", "default_dtype", "float32_matmul_precision")
        ) or any(
            type(environment[k]) is not bool
            for k in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32")
        ):
            raise ValueError("Invalid runtime/precision requirements")
        if not object_value(manifest["network"], "network"):
            raise ValueError("Resolved network configuration required")
    return manifest


def face_identity(face: Json) -> tuple[str, ...]:
    fields = ("field", "kind", "axis", "side")
    if not all(isinstance(face.get(k), str) and face[k] for k in fields):
        raise ValueError("Boundary identity requires field/kind/axis/side")
    if face["side"] not in ("lower", "upper"):
        raise ValueError("Unknown boundary side")
    return tuple(face[k] for k in fields)
