"""Compact convergence reports without changing models or running CUDA."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .schema import Json, digest, identity, inside, number, object_value


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


def compact_snapshot(raw: Json, limits: Json) -> Json:
    metrics = object_value(raw.get("metrics"), "snapshot metrics")
    compact = {key: number(value, key) for key, value in metrics.items()}
    if not {"initial_rmse", *limits.keys()} <= compact.keys():
        raise ValueError("Snapshot is missing required physical metrics")
    operators = raw.get("boundary_operators")
    if not isinstance(operators, list) or not operators:
        raise ValueError("Snapshot boundary operators are missing")
    maximum = max(
        number(object_value(face, "face").get("max_error"), "face max_error") for face in operators
    )
    step = raw.get("lb_steps")
    if type(step) is not int or step < 0:
        raise ValueError("Snapshot lb_steps must be a nonnegative integer")
    return {
        "lb_steps": step,
        "seconds": number(raw.get("seconds"), "snapshot seconds"),
        "closures": number(raw.get("closures"), "closures"),
        "metrics": compact,
        "boundary_max": maximum,
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
        compact_snapshot(object_value(s, "snapshot"), limits) for s in raw.get("snapshots", [])
    ]
    if [s["lb_steps"] for s in snapshots] != manifest["report_parameters"]["lbfgs_snapshots"]:
        raise ValueError("Trial snapshots do not match the complete declared schedule")
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
    for key in ("initial_state_sha256", "fixed_objective_sha256"):
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
        if raw.get(key) != manifest["report_parameters"][key]:
            raise ValueError(f"Trial {key} differs from declared protocol")
    compact.update(
        adam_steps=raw["adam_steps"],
        lbfgs_tolerance_change=raw["lbfgs_tolerance_change"],
        snapshots=snapshots,
        crossing=crossing(snapshots),
    )
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
