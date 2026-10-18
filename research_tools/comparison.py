"""Explicit paired gates; absolute accuracy is separate from preservation."""

from __future__ import annotations

import statistics
from pathlib import Path

from .evidence import compact_trial
from .schema import Json, arm_fields, identity, number, object_value, validate_manifest


def compare(evidence: Json) -> Json:
    if evidence.get("kind") == "frozen-profile-evidence":
        from .frozen_profile import compare as compare_profile

        return compare_profile(evidence)
    if evidence.get("contract") != 1 or evidence.get("kind") != "research-evidence":
        raise ValueError("Expected research evidence contract 1")
    manifest = validate_manifest(object_value(evidence["manifest"], "manifest"), Path.cwd())
    environment = object_value(evidence["environment"], "environment")
    if environment.get("revision") != manifest["revision"]:
        raise ValueError("Evidence revision differs from manifest")
    if (
        manifest["registration"] == "prospective"
        and environment.get("working_tree_dirty") is not False
    ):
        raise ValueError("Prospective evidence requires explicitly clean source")
    if any(
        environment.get(k) != v for k, v in manifest.get("environment_requirements", {}).items()
    ):
        raise ValueError("Runtime/precision differs from registered protocol")
    trials: list[Json] = []
    for raw in evidence["trials"]:
        trial = object_value(raw, "trial")
        snapshots = []
        for value in trial["snapshots"]:
            snap = object_value(value, "snapshot")
            snapshots.append(
                snap
                | {
                    "boundary_operators": snap.get("boundary_operators")
                    or [{"max_error": snap["boundary_max"]}]
                }
            )
        # Revalidate cached values and recompute crossings; never trust saved pass flags.
        trials.append(compact_trial(trial | {"snapshots": snapshots}, manifest))
    keys = [identity(t) for t in trials]
    if len(keys) != len(set(keys)) or set(keys) != {
        identity(t) for t in manifest["expected_trials"]
    }:
        raise ValueError("Evidence does not contain the declared complete matrix")
    settings = manifest["comparison"]
    results: list[Json] = []
    for case in manifest["case_limits"]:
        paired: list[Json] = []
        limits = manifest["case_limits"][case]
        seeds = sorted({t["seed"] for t in trials if t["case"] == case})
        for seed in seeds:
            members = [t for t in trials if t["case"] == case and t["seed"] == seed]
            baseline, candidate = (
                next(
                    t
                    for t in members
                    if all(t[k] == settings[name][k] for k in arm_fields(manifest))
                )
                for name in ("baseline", "candidate")
            )
            fingerprint = (
                "objective_prefix_sha256" if "density" in manifest else "fixed_objective_sha256"
            )
            for key in (
                "initial_state_sha256",
                fingerprint,
                "prediction_points",
                "prediction_batch_size",
                "prediction_warmup_passes",
                "prediction_measurement_passes",
            ):
                if baseline[key] != candidate[key]:
                    raise ValueError(f"{case}/{seed}: incomparable pair ({key})")
            if (
                "density" in manifest
                and baseline["fixed_objective_sha256"] != candidate["objective_prefix_sha256"]
            ):
                raise ValueError(f"{case}/{seed}: baseline is not the candidate prefix")
            physical = True
            for trial in members:
                for snap in trial["snapshots"]:
                    physical &= (
                        number(snap["metrics"]["initial_rmse"], "initial_rmse")
                        <= manifest["physical_limits"]["initial_rmse"]
                    )
                    physical &= (
                        number(snap["boundary_max"], "boundary_max")
                        <= manifest["physical_limits"]["boundary_max"]
                    )
                    for key in limits:
                        number(snap["metrics"][key], key)
            left = baseline["snapshots"][-1]["metrics"]
            right = candidate["snapshots"][-1]["metrics"]
            quality = all(
                right[k] <= max(settings["quality_factor"] * left[k], limit)
                for k, limit in limits.items()
            )
            preserved = (
                baseline["crossing"]["status"] != "observed"
                or candidate["crossing"]["status"] == "observed"
            )
            paired.append(
                {
                    "seed": seed,
                    "physical_passed": bool(physical),
                    "paired_quality_passed": quality,
                    "confirmed_crossing_preserved": preserved,
                    "baseline_final_target_passed": all(
                        left[k] <= limit for k, limit in limits.items()
                    ),
                    "candidate_final_target_passed": all(
                        right[k] <= limit for k, limit in limits.items()
                    ),
                    "baseline_crossing": baseline["crossing"],
                    "candidate_crossing": candidate["crossing"],
                    "training_ratio": number(
                        candidate["training_seconds_with_snapshots"],
                        "candidate time",
                        positive=True,
                    )
                    / number(
                        baseline["training_seconds_with_snapshots"], "baseline time", positive=True
                    ),
                    "prediction_ratio": number(
                        candidate["warm_prediction_seconds"], "candidate prediction", positive=True
                    )
                    / number(
                        baseline["warm_prediction_seconds"], "baseline prediction", positive=True
                    ),
                }
            )
        ratios = [p["training_ratio"] for p in paired]
        median, worst = statistics.median(ratios), max(ratios)
        results.append(
            {
                "case": case,
                "pairs": len(paired),
                "physical_passed": all(p["physical_passed"] for p in paired),
                "paired_quality_passed": all(p["paired_quality_passed"] for p in paired),
                "confirmed_crossing_preserved": all(
                    p["confirmed_crossing_preserved"] for p in paired
                ),
                "candidate_final_target_passed": sum(
                    p["candidate_final_target_passed"] for p in paired
                ),
                "candidate_confirmed": sum(
                    p["candidate_crossing"]["status"] == "observed" for p in paired
                ),
                "median_training_ratio": median,
                "worst_training_ratio": worst,
                "timing_passed": median <= settings["median_training_ratio_max"]
                and worst <= settings["worst_training_ratio_max"],
                "details": paired,
            }
        )
        if "density" in manifest:
            results[-1]["density_target_passed"] = results[-1]["physical_passed"] and results[-1][
                "candidate_confirmed"
            ] == len(paired)
    passed = all(
        all(
            case[key]
            for key in (
                "physical_passed",
                "paired_quality_passed",
                "confirmed_crossing_preserved",
                "timing_passed",
            )
        )
        for case in results
    )
    if "density" in manifest:
        passed = all(case["density_target_passed"] for case in results)
    return {
        "contract": 1,
        "kind": "research-comparison",
        "registration": manifest["registration"],
        "revision": manifest["revision"],
        "trials": len(trials),
        "comparison_passed": passed,
        "working_tree_dirty": evidence["environment"].get("working_tree_dirty"),
        "cases": results,
        "limitations": "Paired gates do not imply absolute target accuracy or authorize default promotion. Crossing times are sampled intervals. Dirty source requires separate provenance review.",
    }
