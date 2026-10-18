"""Bounded full-step timing using the existing immutable research queue."""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
from pathlib import Path

from .frozen_profile import validate as validate_profile
from .frozen_timing import SOURCE_FIELDS, expected
from .schema import Json, digest, inside, number, object_value, read_object

FIELDS = {
    "contract",
    "kind",
    "registration",
    "revision",
    "sources",
    "repeats",
    "points",
    "steps",
    "environment_requirements",
    "comparison",
}
GATES = dict(
    median_total_ratio_max=0.95,
    worst_total_ratio_max=1.10,
    final_pde_rmse_max=0.003,
    final_relative_l2_max=0.002,
    pair_field_max_difference=5e-5,
)


def validate(manifest: Json) -> Json:
    if (
        set(manifest) != FIELDS
        or manifest["contract"] != 5
        or manifest["kind"] != "lbfgs-timing"
        or manifest["registration"] != "prospective"
    ):
        raise ValueError("Expected prospective lbfgs-timing contract 5")
    if (
        manifest["repeats"] != 3
        or type(manifest["repeats"]) is not int
        or manifest["points"] != 1024
        or type(manifest["points"]) is not int
        or manifest["steps"] != 10
        or type(manifest["steps"]) is not int
        or manifest["comparison"] != GATES
    ):
        raise ValueError("Unsupported L-BFGS timing budget/gates")
    sources = manifest["sources"]
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("Two saved sources required")
    for source, seed in zip(sources, (19, 4321)):
        if (
            set(source) != SOURCE_FIELDS
            or source["case"] != "heat_neumann_polynomial3d"
            or type(source["seed"]) is not int
            or source["seed"] != seed
        ):
            raise ValueError("Expected polynomial 3D sources 19/4321")
        validate_profile(
            dict(
                contract=3,
                kind="frozen-profile",
                registration="prospective",
                revision=manifest["revision"],
                case="heat_neumann_polynomial2d",
                points=1024,
                environment_requirements=manifest["environment_requirements"],
                **{k: source[k] for k in ("trial", "weights", "weights_sha256", "config_sha256")},
            )
        )
    return manifest


def plan(manifest: Json) -> Json:
    validate(manifest)
    return dict(
        contract=1,
        kind="lbfgs-timing-plan",
        processes=12,
        measured_outer_steps=240,
        warmup_outer_steps=12,
        points=1024,
        max_iter=1,
        line_search="strong_wolfe",
        source_weights_modified=False,
        timing_modes=["total", "attributed"],
        revision=manifest["revision"],
    )


def compare(evidence: Json) -> Json:
    manifest = validate(object_value(evidence["manifest"], "manifest"))
    trials = evidence["trials"]
    if evidence.get("kind") != "lbfgs-timing-evidence" or evidence.get("contract") != 1:
        raise ValueError("Expected L-BFGS timing evidence")
    if [(t["source_index"], t["repeat"], t["rule"]) for t in trials] != expected(manifest):
        raise ValueError("Incomplete L-BFGS matrix")
    for t in trials:
        if (
            t["source"] != manifest["sources"][t["source_index"]]
            or t["revision"] != manifest["revision"]
            or t["environment"] != manifest["environment_requirements"]
            or t["source_unchanged"] is not True
            or t["restored"] is not True
            or t["timing_paths_match"] is not True
        ):
            raise ValueError("L-BFGS provenance or restoration differs")
        if (
            t["final_state_sha256"]["total"] != t["final_state_sha256"]["attributed"]
            or t["restored_state_sha256"] != t["initial_state_sha256"]
        ):
            raise ValueError("L-BFGS state fingerprints differ")
        total, attributed = t["total"], t["attributed"]
        for timing in (total, attributed):
            if timing["steps"] != 10 or timing["closure_calls"] < 10:
                raise ValueError("Invalid optimizer budget")
            if number(timing["seconds"], "seconds") <= 0:
                raise ValueError("Invalid timing")
        rows = attributed["rows"]
        if len(rows) != 10 or sum(r["closure_calls"] for r in rows) != attributed["closure_calls"]:
            raise ValueError("Invalid closure accounting")
        for row in rows:
            elapsed = number(row["seconds"], "step seconds")
            closure = number(row["closure_seconds"], "closure seconds")
            if (
                not 0 <= closure <= elapsed
                or row["optimizer_evaluations"] != row["closure_calls"]
                or abs(row["outside_closure_seconds"] - (elapsed - closure)) > 1e-12
            ):
                raise ValueError("Invalid step decomposition")
        if (
            abs(sum(r["seconds"] for r in rows) - attributed["seconds"]) > 1e-12
            or abs(sum(r["closure_seconds"] for r in rows) - attributed["closure_seconds"]) > 1e-12
            or total["closure_calls"] != attributed["closure_calls"]
        ):
            raise ValueError("Timing path counters differ")
        if len(t["field"]) != 257 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in t["field"]
        ):
            raise ValueError("Missing independent field holdout")
        for key in ("pde_rmse", "relative_l2"):
            number(t[key], key)
        for control in (t["initial_control"], t["final_control"]):
            number(control["loss_absolute_difference"], "loss difference")
            relative = number(control["gradient_relative_l2_difference"], "gradient")
            maximum = number(control["gradient_max_absolute_difference"], "gradient")
            if relative > 2e-4 and maximum > 2e-7:
                raise ValueError("Independent gradient control failed")
    ratios, differences = [], []
    quality = True
    for i in range(0, len(trials), 2):
        pair = {t["rule"]: t for t in trials[i : i + 2]}
        a, b = pair["autograd"], pair["analytic"]
        if any(
            a[k] != b[k] for k in ("initial_state_sha256", "points_sha256", "initial_field_sha256")
        ):
            raise ValueError("Initial pair differs")
        ratio = b["total"]["seconds"] / a["total"]["seconds"]
        difference = max(abs(x - y) for x, y in zip(a["field"], b["field"]))
        ratios.append(ratio)
        differences.append(difference)
        quality &= difference <= GATES["pair_field_max_difference"] and all(
            t["pde_rmse"] <= GATES["final_pde_rmse_max"]
            and t["relative_l2"] <= GATES["final_relative_l2_max"]
            for t in (a, b)
        )
    median, worst = statistics.median(ratios), max(ratios)
    return dict(
        contract=1,
        kind="lbfgs-timing-comparison",
        trials=len(trials),
        median_total_ratio=median,
        worst_total_ratio=worst,
        numerical_passed=quality,
        pair_field_max_difference=max(differences),
        comparison_passed=quality
        and median <= GATES["median_total_ratio_max"]
        and worst <= GATES["worst_total_ratio_max"],
        details=dict(pairs=ratios),
        scope="One case/two saved states, not time to target",
    )


def collect(root: Path, manifest: Json) -> Json:
    evidence = read_object(root / "lbfgs.json")
    if evidence["manifest"] != validate(manifest):
        raise ValueError("L-BFGS manifest differs")
    artifacts = evidence["artifact_sha256"]
    labels = [f"s{i}-r{r}-{rule}" for i, r, rule in expected(manifest)]
    if set(artifacts) != {label + suffix for label in labels for suffix in (".json", ".log")}:
        raise ValueError("Incomplete L-BFGS artifact inventory")
    for relative, checksum in artifacts.items():
        if digest(inside(root, relative)) != checksum:
            raise ValueError("L-BFGS artifact checksum differs")
    if [read_object(root / (label + ".json")) for label in labels] != evidence["trials"]:
        raise ValueError("L-BFGS summary differs from child reports")
    compare(evidence)
    return evidence


def run(root: Path, checkout: Path, output: Path, manifest: Json) -> None:
    validate(manifest)
    if Path.cwd() != root or str(root) != r"C:\Projects\pinn-runner":
        raise ValueError("Guarded Windows root required")
    output.mkdir(parents=True, exist_ok=False)
    trials, artifacts = [], {}
    for index, repeat, rule in expected(manifest):
        label = f"s{index}-r{repeat}-{rule}"
        destination = output / (label + ".json")
        with (output / (label + ".log")).open("wb") as log:
            subprocess.run(
                [
                    str(root / ".venv/Scripts/python.exe"),
                    "-B",
                    str(checkout / "scripts/research_lbfgs.py"),
                    "--manifest",
                    str(output.parent / "manifest.json"),
                    "--output",
                    str(destination),
                    "--source-index",
                    str(index),
                    "--repeat",
                    str(repeat),
                    "--rule",
                    rule,
                ],
                cwd=root,
                env=dict(os.environ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        trials.append(read_object(destination))
        artifacts[destination.name] = digest(destination)
        artifacts[label + ".log"] = digest(output / (label + ".log"))
        progress = dict(completed_trials=len(trials), expected_trials=12)
        (output.parent / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
        print(json.dumps(progress), flush=True)
    evidence = dict(
        contract=1,
        kind="lbfgs-timing-evidence",
        manifest=manifest,
        trials=trials,
        artifact_sha256=artifacts,
    )
    compare(evidence)
    (output / "lbfgs.json").write_text(json.dumps(evidence, allow_nan=False), encoding="utf-8")
