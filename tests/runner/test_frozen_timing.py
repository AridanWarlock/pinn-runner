"""Frozen timing gates revalidate raw samples and exact pair provenance."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from research_tools import frozen_timing
from research_tools.comparison import compare
from research_tools.evidence import read_bundle, write_bundle
from tests.runner.test_frozen_profile import protocol as profile_protocol


def protocol() -> dict[str, Any]:
    old = profile_protocol()
    return dict(
        contract=4,
        kind="frozen-timing",
        registration="prospective",
        revision=old["revision"],
        sources=[
            dict(
                case=old["case"],
                seed=19,
                **{k: old[k] for k in ("trial", "weights", "weights_sha256", "config_sha256")},
            )
        ],
        repeats=3,
        points=1024,
        prediction_points=8193,
        blocks=9,
        block_seconds=6.0,
        prediction_block_seconds=0.75,
        environment_requirements=old["environment_requirements"],
        comparison=dict(
            median_objective_ratio_max=0.95,
            worst_objective_ratio_max=1.1,
            median_prediction_ratio_max=1.1,
            worst_allocated_ratio_max=1.1,
        ),
    )


def evidence() -> dict[str, Any]:
    manifest = protocol()
    trials = []
    for i, r, rule in frozen_timing.expected(manifest):
        calls = 1200 if rule == "analytic" else 1000
        objective = dict(
            samples=[dict(seconds=6.0, calls=calls)] * 9,
            median_seconds=6.0 / calls,
            peak_allocated_bytes=100,
            peak_reserved_bytes=100,
        )
        prediction = dict(
            samples=[dict(seconds=0.75, calls=100)] * 9,
            median_seconds=0.75 / 100,
            peak_allocated_bytes=100,
            peak_reserved_bytes=100,
        )
        trials.append(
            dict(
                source_index=i,
                repeat=r,
                rule=rule,
                source=manifest["sources"][i],
                environment=manifest["environment_requirements"],
                revision=manifest["revision"],
                weights_unchanged=True,
                source_unchanged=True,
                profiled=False,
                independent_autograd_control=dict(
                    loss_absolute_difference=0.0,
                    gradient_max_absolute_difference=0.0,
                    gradient_relative_l2_difference=0.0,
                ),
                state_sha256="a" * 64,
                prediction_sha256="b" * 64,
                points_sha256="c" * 64,
                objective=objective,
                prediction={"128": copy.deepcopy(prediction), "2048": copy.deepcopy(prediction)},
                cold_ready_seconds=3.0,
                cold_first_prediction_seconds=0.1,
            )
        )
    return dict(
        contract=1,
        kind="frozen-timing-evidence",
        manifest=manifest,
        trials=trials,
        artifact_sha256={},
    )


def test_timing_budget_and_process_order_are_explicit() -> None:
    result = frozen_timing.plan(protocol())
    assert result["processes"] == 6
    assert result["minimum_measured_seconds"] == 405
    assert result["optimizer_updates"] == 0
    assert frozen_timing.expected(protocol())[2][2] == "analytic"


def test_timing_collectable_bundle_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    write_bundle(output, evidence())
    result = compare(read_bundle(output))
    assert result["comparison_passed"] is True
    assert result["cases"][0]["median_objective_ratio"] == pytest.approx(5 / 6)
    assert result["numerical_accuracy_claim"] is False


@pytest.mark.parametrize(
    "failure", ["missing", "median", "profiled", "weights", "points", "duration"]
)
def test_timing_invalid_evidence_rejected(failure: str) -> None:
    data = evidence()
    t = data["trials"][0]
    if failure == "missing":
        data["trials"].pop()
    elif failure == "median":
        t["objective"]["median_seconds"] = 1.0
    elif failure == "profiled":
        t["profiled"] = True
    elif failure == "weights":
        t["weights_unchanged"] = False
    elif failure == "points":
        t["points_sha256"] = "d" * 64
    elif failure == "duration":
        t["objective"]["samples"][0]["seconds"] = 1.0
    with pytest.raises(ValueError):
        compare(data)


def test_real_slowdown_is_kept_as_negative_result() -> None:
    data = evidence()
    for t in data["trials"]:
        if t["rule"] == "analytic":
            t["objective"]["samples"] = [dict(seconds=6.0, calls=800)] * 9
            t["objective"]["median_seconds"] = 6 / 800
    assert compare(data)["comparison_passed"] is False


@pytest.mark.parametrize(
    "changes", [dict(repeats=4), dict(prediction_points=8192), dict(blocks=True), dict(unknown=1)]
)
def test_timing_unregistered_budget_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        frozen_timing.plan(protocol() | changes)


@pytest.mark.parametrize("details", [False, True])
def test_timing_cli_expands_pairs_only_on_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], details: bool
) -> None:
    import json

    from scripts.research import main

    output = tmp_path / "bundle"
    write_bundle(output, evidence())
    arguments = ["compare", "--bundle", str(output)]
    if details:
        arguments.append("--details")
    assert main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["comparison_passed"] is True
    assert ("details" in result) is details
    assert result["cases"][0]["median_objective_ratio"] == pytest.approx(5 / 6)
