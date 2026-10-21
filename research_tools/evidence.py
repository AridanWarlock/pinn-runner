"""Compact convergence reports without changing models or running CUDA."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .schema import Json, digest, face_identity, identity, inside, number, object_value


def status(root: Path) -> Json:
    source = root / "status.json"
    if not source.is_file():
        raise ValueError("Study status.json is missing; state is unknown")
    value = object_value(json.loads(source.read_text(encoding="utf-8")), "status")
    return compact_status(value)


def compact_status(value: Json) -> Json:
    if not isinstance(value.get("state"), str):
        raise ValueError("Status state is missing")
    return {
        "contract": 1,
        "kind": "research-status",
        **{
            key: value[key]
            for key in (
                "state",
                "commit",
                "started_at",
                "finished_at",
                "updated_at",
                "error",
            )
            if key in value
        },
        "completed_cases": len(value.get("completed_cases", [])),
    }


def compact_snapshot(raw: Json, limits: Json, faces: list[Json] | None = None) -> Json:
    metrics = object_value(raw.get("metrics"), "snapshot metrics")
    compact = {key: number(value, key) for key, value in metrics.items()}
    if not {"initial_rmse", *limits.keys()} <= compact.keys():
        raise ValueError("Snapshot is missing required physical metrics")
    operators = raw.get("boundary_operators")
    if not isinstance(operators, list) or not operators:
        raise ValueError("Snapshot boundary operators are missing")
    compact_operators = []
    if faces is not None:
        ids = [face_identity(object_value(face, "face")) for face in operators]
        if len(ids) != len(set(ids)) or set(ids) != {face_identity(face) for face in faces}:
            raise ValueError("Missing, duplicate or unexpected boundary faces")
        compact_operators = [
            {
                **{k: face[k] for k in ("field", "kind", "axis", "side")},
                "max_error": number(face["max_error"], "face error"),
            }
            for face in operators
        ]
    maximum = max(
        number(object_value(face, "face").get("max_error"), "face max_error") for face in operators
    )
    step = raw.get("lb_steps")
    if type(step) is not int or step < 0:
        raise ValueError("Snapshot lb_steps must be a nonnegative integer")
    return {
        **{
            key: number(raw[key], key)
            for key in ("adam_residual_points", "lbfgs_residual_points")
            if key in raw
        },
        "lb_steps": step,
        "seconds": number(raw.get("seconds"), "snapshot seconds"),
        "closures": number(raw.get("closures"), "closures"),
        "metrics": compact,
        "boundary_max": maximum,
        "boundary_operators": compact_operators,
        "passed": all(compact[key] <= limit for key, limit in limits.items()),
    }


def crossing(snapshots: list[Json]) -> Json:
    for previous, current in zip(snapshots, snapshots[1:]):
        if previous["passed"] and current["passed"]:
            index = snapshots.index(previous)
            return {
                "status": "observed",
                "lower_seconds": snapshots[index - 1]["seconds"] if index else 0.0,
                "upper_seconds": previous["seconds"],
                "confirmed_seconds": current["seconds"],
                "left_censored": index == 0,
            }
    return {"status": "not_confirmed"}


def compact_trial(raw: Json, manifest: Json) -> Json:
    identity(raw)
    limits = manifest["case_limits"].get(raw["case"])
    if limits is None:
        raise ValueError("Unexpected case")
    snapshots = [
        compact_snapshot(
            object_value(s, "snapshot"), limits, manifest.get("case_faces", {}).get(raw["case"])
        )
        for s in raw.get("snapshots", [])
    ]
    if "hypothesis_evaluation" in manifest:
        from .surrogate_evidence import validate_snapshots

        validate_snapshots(raw, snapshots, manifest)
    if "absolute_quality" in manifest:
        for snap in snapshots:
            snap["passed"] &= (
                snap["metrics"]["initial_rmse"] <= manifest["physical_limits"]["initial_rmse"]
                and snap["boundary_max"] <= manifest["physical_limits"]["boundary_max"]
            )
    if "density" not in manifest:
        if [s["lb_steps"] for s in snapshots] != manifest["report_parameters"]["lbfgs_snapshots"]:
            raise ValueError("Trial snapshots do not match the complete declared schedule")
    else:
        density = manifest["density"]
        steps = [s["lb_steps"] for s in snapshots]
        if not steps or steps[0] != 0 or steps != sorted(set(steps)):
            raise ValueError("Density snapshots must start at zero and increase")
        if steps[-1] > manifest["report_parameters"]["lbfgs_snapshots"][-1]:
            raise ValueError("Density exceeded registered step cap")
        if raw.get("density") != density or raw.get("stop_reason") != "time_budget":
            raise ValueError("Density trial did not finish its registered time budget")
        if (
            number(raw.get("training_seconds_with_snapshots"), "training seconds")
            < density["training_seconds"]
        ):
            raise ValueError("Density trial ended before its time budget")
        if (
            type(raw.get("adam_steps")) is not int
            or not 0 < raw["adam_steps"] <= manifest["report_parameters"]["adam_steps"]
        ):
            raise ValueError("Invalid completed Adam budget")
        for snap in snapshots:
            snap["passed"] &= (
                snap["seconds"] <= density["training_seconds"]
                and snap["metrics"]["initial_rmse"] <= manifest["physical_limits"]["initial_rmse"]
                and snap["boundary_max"] <= manifest["physical_limits"]["boundary_max"]
            )
        for original, snap in zip(raw["snapshots"], snapshots):
            for key in ("adam_residual_points", "lbfgs_residual_points"):
                snap[key] = number(original.get(key), key)
            if snap["lbfgs_residual_points"] != snap["closures"] * raw["points_override"]:
                raise ValueError("Density residual counter differs from closure work")
        if any(a["closures"] > b["closures"] for a, b in zip(snapshots, snapshots[1:])):
            raise ValueError("Density closure counters go backwards")
    if any(a["seconds"] > b["seconds"] for a, b in zip(snapshots, snapshots[1:])):
        raise ValueError("Snapshot time goes backwards")
    if (
        raw.get("control_feature_derivatives") != "autograd"
        or raw.get("control_prediction_execution") != "eager"
        or raw.get("control_prediction_background") != "general"
    ):
        raise ValueError("Independent eager/general/autograd evaluation is required")
    if raw.get("feature_derivatives") != (
        "analytic" if raw["backend"] == "neumann_analytic" else "autograd"
    ):
        raise ValueError("Reported feature derivative rule does not match executor")
    compact = {
        key: raw[key]
        for key in (
            "case",
            "seed",
            "backend",
            "lifting",
            "backward",
            "background",
            "spatial_encoding",
            "feature_derivatives",
            "control_feature_derivatives",
            "control_prediction_execution",
            "control_prediction_background",
        )
    }
    if "absolute_quality" in manifest:
        if manifest["report_parameters"]["constraints"] != "soft" or "constraints" in raw:
            if raw.get("constraints") != manifest["report_parameters"]["constraints"]:
                raise ValueError("Actual constraint representation differs from protocol")
            compact["constraints"] = raw["constraints"]
        checksum = raw.get("problem_source_sha256")
        if checksum != manifest["problem_sha256"][raw["case"]]:
            raise ValueError("Problem source differs from absolute-quality protocol")
        compact["problem_source_sha256"] = checksum
        if "layer_evaluation" in manifest:
            if json.dumps(raw.get("layer_evaluation"), sort_keys=True) != json.dumps(
                manifest["layer_evaluation"], sort_keys=True
            ):
                raise ValueError("Actual layer evaluator differs from protocol")
            compact["layer_evaluation"] = raw["layer_evaluation"]
    if "physics" in manifest:
        from benchmarks.conductivity import source_sha256

        representation = raw.get("coefficient_representation")
        if representation not in ("formula", "table"):
            raise ValueError("Missing coefficient representation")
        if raw.get("physics") != manifest["physics"][raw["case"]]:
            raise ValueError("Trial physics differs from protocol")
        expected_source = source_sha256(raw["case"], representation)
        if raw.get("problem_source_sha256") != expected_source:
            raise ValueError("Problem source differs from registered representation")
        compact.update(
            coefficient_representation=representation,
            physics=raw["physics"],
            problem_source_sha256=expected_source,
        )
    elif "coefficient_representation" in raw:
        raise ValueError("Coefficient trials require a physics protocol")
    for key in (
        *(
            ("initial_parameters_sha256",)
            if "physics" in manifest
            or ("absolute_quality" in manifest and "initial_parameters_sha256" in raw)
            else ()
        ),
        "initial_state_sha256",
        "fixed_objective_sha256",
        *(("objective_prefix_sha256",) if "density" in manifest else ()),
    ):
        value = raw.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"Missing/invalid {key}")
        compact[key] = value
    for key in ("training_seconds_with_snapshots", "warm_prediction_seconds"):
        compact[key] = number(raw.get(key), key, positive=True)
    if compact["training_seconds_with_snapshots"] < snapshots[-1]["seconds"]:
        raise ValueError("Training time precedes its last snapshot")
    for key in (
        "prediction_points",
        "prediction_batch_size",
        "prediction_warmup_passes",
        "prediction_measurement_passes",
    ):
        value = raw.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key}: positive integer required")
        compact[key] = value
    for key in ("adam_steps", "lbfgs_tolerance_change"):
        if key == "adam_steps" and "density" in manifest:
            continue
        if raw.get(key) != manifest["report_parameters"][key]:
            raise ValueError(f"Trial {key} differs from declared protocol")
    if manifest["contract"] == 2:
        if raw.get("network") != manifest["network"]:
            raise ValueError("Trial network differs from declared architecture")
        compact["network"] = raw["network"]
    compact.update(
        adam_steps=raw["adam_steps"],
        lbfgs_tolerance_change=raw["lbfgs_tolerance_change"],
        snapshots=snapshots,
        crossing=crossing(snapshots),
    )
    for key in (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "snapshot_seconds",
        "setup_seconds",
        "evaluation_seconds",
        "trial_wall_seconds",
        "process_seconds",
    ):
        if raw.get(key) is not None:
            compact[key] = number(raw[key], key)
    if "density" in manifest:
        compact.update(
            {
                key: raw[key]
                for key in (
                    "points_override",
                    "density",
                    "stop_reason",
                    "budget_overshoot_seconds",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "density_preparation_seconds",
                )
            }
        )
        budget = manifest["density"]["training_seconds"]
        if (
            abs(
                compact["budget_overshoot_seconds"]
                - (compact["training_seconds_with_snapshots"] - budget)
            )
            > 1e-6
        ):
            raise ValueError("Density overshoot differs from measured training time")
        for key in (
            "budget_overshoot_seconds",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "density_preparation_seconds",
        ):
            number(compact[key], key)
    return compact


def collect(root: Path, manifest: Json, manifest_path: Path) -> Json:
    trials: list[Json] = []
    sources: list[Json] = []
    environment: Json | None = None
    for relative in manifest["reports"]:
        path = inside(root, relative)
        source = path.read_bytes()
        report = object_value(json.loads(source.decode("utf-8")), relative)
        if report.get("contract") != 3 or report.get("status") != "completed":
            raise ValueError(f"{relative}: only completed convergence contract 3 accepted")
        case = report.get("case")
        if report.get("limits") != manifest["case_limits"].get(case):
            raise ValueError("Report quality limits differ from declared protocol")
        if any(report.get(key) != value for key, value in manifest["report_parameters"].items()):
            raise ValueError("Report budget/configuration differs from declared protocol")
        if "density" in manifest and report.get("density") != manifest["density"]:
            raise ValueError("Report density budget differs from declared protocol")
        current = object_value(report.get("environment"), "environment")
        if (
            manifest["registration"] == "prospective"
            and current.get("working_tree_dirty") is not False
        ):
            raise ValueError("Prospective evidence requires explicitly clean source")
        if current.get("revision") != manifest["revision"]:
            raise ValueError("Report source revision differs from manifest")
        if environment is not None and current != environment:
            raise ValueError("Environment differs across reports")
        if any(
            current.get(k) != v for k, v in manifest.get("environment_requirements", {}).items()
        ):
            raise ValueError("Runtime/precision differs from declared protocol")
        environment = current
        raw_trials = report.get("trials")
        if not isinstance(raw_trials, list) or not raw_trials:
            raise ValueError("Report has no trials")
        for trial in raw_trials:
            value = object_value(trial, "trial")
            if value.get("case") != case:
                raise ValueError("Trial case differs from report")
            trials.append(compact_trial(value, manifest))
        sources.append({"path": relative, "sha256": hashlib.sha256(source).hexdigest()})
    actual = [identity(t) for t in trials]
    expected = {identity(t) for t in manifest["expected_trials"]}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Missing, duplicate or unexpected trials; collection is incomplete")
    return {
        "contract": 1,
        "kind": "research-evidence",
        "manifest": manifest,
        "manifest_sha256": digest(manifest_path),
        "sources": sources,
        "environment": environment,
        "trials": trials,
    }


def write_bundle(output: Path, evidence: Json) -> None:
    # Refuse overwrites: an earlier comparison must remain reproducible.
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    rows: list[Json] = []
    snapshots: list[Json] = []
    for trial in evidence["trials"]:
        if evidence.get("kind") == "public-prediction-evidence":
            rows.append(trial)
            continue
        if evidence.get("kind") == "allocation-profile-evidence":
            rows.extend(dict(rule=trial["rule"], **row) for row in trial["profiling"]["stages"])
            continue
        if evidence.get("kind") == "numerical-acceptance-evidence":
            rows.append(dict(nodeid=trial["nodeid"], **trial["phases"], seconds=trial["seconds"]))
            continue
        if evidence.get("kind") == "lbfgs-timing-evidence":
            rows.append(
                dict(
                    source_index=trial["source_index"],
                    repeat=trial["repeat"],
                    rule=trial["rule"],
                    case=trial["source"]["case"],
                    seed=trial["source"]["seed"],
                    total_seconds=trial["total"]["seconds"],
                    closure_calls=trial["total"]["closure_calls"],
                    attributed_seconds=trial["attributed"]["seconds"],
                    attributed_closure_seconds=trial["attributed"]["closure_seconds"],
                    pde_rmse=trial["pde_rmse"],
                    relative_l2=trial["relative_l2"],
                )
            )
            continue
        if evidence.get("kind") == "frozen-timing-evidence":
            rows.append(
                dict(
                    source_index=trial["source_index"],
                    repeat=trial["repeat"],
                    rule=trial["rule"],
                    case=trial["source"]["case"],
                    seed=trial["source"]["seed"],
                    objective_seconds=trial["objective"]["median_seconds"],
                    prediction128_seconds=trial["prediction"]["128"]["median_seconds"],
                    prediction2048_seconds=trial["prediction"]["2048"]["median_seconds"],
                    peak_allocated_bytes=trial["objective"]["peak_allocated_bytes"],
                    peak_reserved_bytes=trial["objective"]["peak_reserved_bytes"],
                    cold_ready_seconds=trial["cold_ready_seconds"],
                    cold_first_prediction_seconds=trial["cold_first_prediction_seconds"],
                )
            )
            continue
        if evidence.get("kind") == "frozen-profile-evidence":
            rows.append(
                dict(
                    rule=trial["rule"],
                    profile_exit_code=trial["profile_exit_code"],
                    kernel_count=trial["trace"].get("kernels", {}).get("count", 0),
                )
            )
            continue
        common = {
            key: trial[key]
            for key in (
                "case",
                "seed",
                "backend",
                "lifting",
                "backward",
                "background",
                "spatial_encoding",
            )
        }
        if "physics" in evidence["manifest"]:
            common["coefficient_representation"] = trial["coefficient_representation"]
        if "density" in evidence["manifest"]:
            common["points_override"] = trial["points_override"]
        rows.append(
            common
            | trial["snapshots"][-1]["metrics"]
            | {
                key: trial[key]
                for key in ("training_seconds_with_snapshots", "warm_prediction_seconds")
            }
            | {
                "boundary_max": trial["snapshots"][-1]["boundary_max"],
                "crossing_status": trial["crossing"]["status"],
            }
        )
        for snap in trial["snapshots"]:
            snapshots.append(
                common
                | {
                    key: snap[key]
                    for key in ("lb_steps", "seconds", "closures", "boundary_max", "passed")
                }
                | snap["metrics"]
            )
    for filename, values in (("final.csv", rows), ("snapshots.csv", snapshots)):
        fields = list(dict.fromkeys(key for row in values for key in row))
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)

    completion = {
        "contract": 1,
        "kind": "research-bundle",
        "sha256": {name: digest(output / name) for name in BUNDLE_FILES},
    }
    temporary = output / "bundle.json.tmp"
    temporary.write_text(json.dumps(completion, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "bundle.json")


BUNDLE_FILES = ("evidence.json", "final.csv", "snapshots.csv")


def read_bundle(directory: Path) -> Json:
    marker = inside(directory, "bundle.json")
    if not marker.is_file():
        raise ValueError(
            "Incomplete bundle: completion record is missing; recollect to a new directory"
        )
    record = object_value(json.loads(marker.read_text(encoding="utf-8")), "bundle")
    if (
        set(record) != {"contract", "kind", "sha256"}
        or record.get("contract") != 1
        or record.get("kind") != "research-bundle"
        or not isinstance(record.get("sha256"), dict)
        or set(record["sha256"]) != set(BUNDLE_FILES)
    ):
        raise ValueError("Invalid bundle completion record")
    evidence_bytes = b""
    for filename in BUNDLE_FILES:
        path = inside(directory, filename)
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != record["sha256"][filename]:
            raise ValueError(f"Bundle checksum differs: {filename}")
        if filename == "evidence.json":
            evidence_bytes = source
    return object_value(json.loads(evidence_bytes), "evidence")
