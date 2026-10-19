"""Isolate each numerical trial without initializing CUDA in the coordinator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pinnstorch.runner.settings import Refinement

from .convergence import ConvergenceArm, ConvergencePlan


def trial_command(
    script: Path, plan: ConvergencePlan, arm: ConvergenceArm, seed: int, output: Path
) -> list[str]:
    command = [
        sys.executable,
        str(script),
        "--shared-process",
        "--output",
        str(output),
        "--case",
        plan.case.name,
        "--seeds",
        str(seed),
        "--steps",
        str(plan.steps),
        "--snapshots",
        *(str(step) for step in plan.snapshots),
        "--constraints",
        plan.constraints,
        "--hard-execution",
        arm.execution,
        "--hard-lifting",
        arm.lifting,
        "--backward",
        arm.backward,
        "--hard-background",
        arm.background,
        "--spatial-encoding",
        arm.encoding,
        "--lbfgs-tolerance-change",
        str(plan.tolerance_change),
    ]
    if arm.coefficient_representation is not None:
        command.extend(["--coefficient-representation", arm.coefficient_representation])
    for name, limit in plan.limits.items():
        command.extend(["--" + name.replace("_", "-"), str(limit)])
    points = plan.points if arm.points is None else arm.points
    if points is not None:
        command.extend(["--points", str(points)])
    if plan.density is not None:
        command.extend(["--density", json.dumps(plan.density)])
    return command


def read_trial(
    directory: Path, plan: ConvergencePlan, arm: ConvergenceArm, seed: int
) -> dict[str, Any]:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    expected = {
        "contract": 3,
        "case": plan.case.name,
        "limits": plan.limits,
        "adam_steps": plan.steps,
        "lbfgs_snapshots": list(plan.snapshots),
        "constraints": plan.constraints,
        "points_override": plan.points if arm.points is None else arm.points,
        "execution_lifecycle": "single_trial",
        "status": "completed",
    }
    if not isinstance(report, dict) or any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("Child report does not match the requested convergence protocol")
    if plan.density is not None and report.get("density") != plan.density:
        raise ValueError("Child density budget differs from protocol")
    if (
        report.get("lbfgs_tolerance_change", Refinement().tolerance_change)
        != plan.tolerance_change
    ):
        raise ValueError("Child report uses a different L-BFGS tolerance_change")
    trials = report.get("trials")
    if not isinstance(trials, list) or len(trials) != 1 or not isinstance(trials[0], dict):
        raise ValueError("Each child must return exactly one trial")
    trial = trials[0]
    identity = {
        "case": plan.case.name,
        "seed": seed,
        "backend": arm.execution,
        "lifting": arm.lifting,
        "backward": arm.backward,
        "background": arm.background,
    }
    if any(trial.get(k) != v for k, v in identity.items()):
        raise ValueError("Child returned a different seed or experiment arm")
    if trial.get("spatial_encoding", "identity") != arm.encoding:
        raise ValueError("Child returned a different spatial representation")
    if trial.get("coefficient_representation") != arm.coefficient_representation:
        raise ValueError("Child returned a different coefficient representation")
    if arm.execution == "neumann_analytic" and trial.get("feature_derivatives") != "analytic":
        raise ValueError("Child report did not execute analytic feature derivatives")
    if not isinstance(report.get("environment"), dict) or not report["environment"].get(
        "revision"
    ):
        raise ValueError("Child report requires a recorded source environment")
    return report


def write_report(output: Path, report: dict[str, Any]) -> None:
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "report.json")


def run_isolated(
    plan: ConvergencePlan,
    arms: list[ConvergenceArm],
    seeds: list[int],
    output: Path,
    script: Path,
) -> dict[str, Any]:
    if not arms or not seeds:
        raise ValueError("At least one arm and seed required")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    combined: dict[str, Any] | None = None
    for index, seed in enumerate(seeds):
        for arm in arms if index % 2 == 0 else list(reversed(arms)):
            child = output / "processes" / f"{arm.name}-{seed}"
            command = trial_command(script, plan, arm, seed, child)
            process_started = time.perf_counter()
            working_directory = Path(
                os.environ.get("PINN_WORKING_DIRECTORY", script.parent.parent)
            )
            subprocess.run(command, cwd=working_directory, check=True)
            process_seconds = time.perf_counter() - process_started
            report = read_trial(child, plan, arm, seed)
            if combined is None:
                combined = {
                    **report,
                    "execution_lifecycle": "isolated_trials",
                    "status": "running",
                    "trials": [],
                }
                if plan.density is not None:
                    combined["points_override"] = plan.points
            elif report["environment"] != combined["environment"]:
                raise ValueError(
                    "Child source revision or runtime environment changed during study"
                )
            trial = dict(report["trials"][0])
            trial.update(
                process_seconds=process_seconds, process_report=str(child / "report.json")
            )
            combined["trials"].append(trial)
            combined["study_seconds"] = time.perf_counter() - started
            write_report(output, combined)
    assert combined is not None
    combined["status"] = "completed"
    write_report(output, combined)
    return combined
