"""Single-configuration accuracy is independent of speed and paired preservation."""

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from benchmarks.catalog import CATALOG
from research_tools.comparison import compare
from research_tools.evidence import compact_trial, read_bundle, write_bundle
from research_tools.planning import plan
from research_tools.schema import read_object

ROOT = Path(__file__).resolve().parents[2]


def manifest() -> dict[str, Any]:
    value = read_object(ROOT / "docs/experiments/2026-10-17-queue-probe-protocol.json")
    arm = dict(
        backend="eager",
        lifting="recursive",
        backward="all",
        background="general",
        spatial_encoding="identity",
    )
    del value["comparison"]
    value.update(
        absolute_quality=arm,
        reports=["advection/report.json", "burgers/report.json"],
        expected_trials=[
            dict(case=case, seed=seed, **arm)
            for case in ("advection", "burgers")
            for seed in (19, 73)
        ],
        case_limits={
            case: dict(
                relative_l2=0.002,
                final_relative_l2=0.003,
                worst_time_relative_l2=0.005,
                pde_rmse=0.003,
            )
            for case in ("advection", "burgers")
        },
        physical_limits=dict(initial_rmse=0.001, boundary_max=0.001),
        case_faces={case: CATALOG[case].faces() for case in ("advection", "burgers")},
        problem_sha256={
            case: hashlib.sha256(
                (ROOT / "benchmarks/problems" / case / "functions.py")
                .read_bytes()
                .replace(b"\r\n", b"\n")
            ).hexdigest()
            for case in ("advection", "burgers")
        },
        network=dict(hidden_layers=[32, 32, 32], spatial_encoding="identity"),
    )
    value["report_parameters"].update(constraints="soft")
    return value


def test_absolute_plan_counts_once_and_requires_all_metrics_and_real_faces() -> None:
    m = manifest()
    p = plan(m)
    assert p["trials"] == 4 and p["adam_steps_total"] == 4
    assert all(case["pairs"] == 0 and case["trials"] == 2 for case in p["cases"])
    assert p["comparison_purpose"] == "absolute_accuracy"
    for damage in ("pair", "source", "outflow", "hard", "temporal"):
        changed = deepcopy(m)
        if damage == "pair":
            changed["expected_trials"].append(deepcopy(changed["expected_trials"][0]))
        elif damage == "source":
            changed["problem_sha256"]["advection"] = "0" * 64
        elif damage == "outflow":
            changed["case_faces"]["advection"].append(
                dict(field="u", kind="dirichlet", axis="x", side="upper")
            )
        elif damage == "hard":
            changed["report_parameters"]["constraints"] = "hard_initial_dirichlet"
        else:
            del changed["case_limits"]["advection"]["worst_time_relative_l2"]
        with pytest.raises(ValueError):
            plan(changed)


def test_absolute_comparison_keeps_negative_quality_and_rechecks_every_gate(
    tmp_path: Path,
) -> None:
    m = manifest()
    trials = []
    for identity in m["expected_trials"]:
        trials.append(
            dict(
                **identity,
                feature_derivatives="autograd",
                control_feature_derivatives="autograd",
                control_prediction_execution="eager",
                control_prediction_background="general",
                problem_source_sha256=m["problem_sha256"][identity["case"]],
                network=m["network"],
                initial_state_sha256="b" * 64,
                fixed_objective_sha256="c" * 64,
                training_seconds_with_snapshots=3,
                warm_prediction_seconds=0.01,
                prediction_points=1271,
                prediction_batch_size=2048,
                prediction_warmup_passes=3,
                prediction_measurement_passes=7,
                adam_steps=1,
                lbfgs_tolerance_change=1e-12,
                snapshots=[
                    dict(
                        lb_steps=step,
                        seconds=step + 1,
                        closures=step,
                        metrics=dict(
                            relative_l2=0.0001,
                            final_relative_l2=0.0002,
                            worst_time_relative_l2=0.0003,
                            pde_rmse=0.0004,
                            initial_rmse=0.0001,
                        ),
                        boundary_operators=[
                            dict(**face, max_error=0.0001)
                            for face in m["case_faces"][identity["case"]]
                        ],
                    )
                    for step in (0, 1)
                ],
            )
        )
    e = dict(
        contract=1,
        kind="research-evidence",
        manifest=m,
        environment=dict(
            revision=m["revision"], working_tree_dirty=False, **m["environment_requirements"]
        ),
        trials=trials,
    )
    assert compare(e)["passed"] == 4 and compare(e)["timing_passed"] is None
    for metric in (*m["case_limits"]["advection"], "initial_rmse", "boundary_max"):
        changed = deepcopy(e)
        last = changed["trials"][0]["snapshots"][-1]
        if metric == "boundary_max":
            last["boundary_operators"][0]["max_error"] = 0.1
        else:
            last["metrics"][metric] = 0.1
        result = compare(changed)
        assert result["passed"] == 3 and not result["comparison_passed"]
        assert metric in result["results"][0]["failed_metrics"]
        assert result["results"][0]["crossing"]["status"] == "not_confirmed"
    # A poor early soft-boundary snapshot does not invalidate a passing final field.
    e["trials"][0]["snapshots"][0]["metrics"]["initial_rmse"] = 0.5
    assert compare(e)["comparison_passed"]
    e["trials"] = [compact_trial(t, m) for t in e["trials"]]
    write_bundle(tmp_path / "bundle", e)
    assert compare(read_bundle(tmp_path / "bundle"))["passed"] == 4
    broken = deepcopy(e)
    broken["trials"].pop()
    with pytest.raises(ValueError, match="complete matrix"):
        compare(broken)
    broken = deepcopy(e)
    broken["trials"][0]["problem_source_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="Problem source"):
        compare(broken)
