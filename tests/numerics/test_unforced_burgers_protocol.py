"""The new evaluation cannot be omitted, weakened or confused with the old case."""

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.burgers_quality import evaluation_identity
from benchmarks.catalog import CATALOG, UNFORCED_VISCOSITIES
from research_tools.planning import plan
from tests.numerics.test_absolute_quality import manifest

ROOT = Path(__file__).resolve().parents[2]


def unforced_manifest():
    m = manifest()
    names = list(UNFORCED_VISCOSITIES)
    arm = m["absolute_quality"]
    m.update(
        reports=[f"{name}/report.json" for name in names],
        expected_trials=[dict(case=name, seed=11, **arm) for name in names],
        case_limits={
            name: dict(
                relative_l2=0.002,
                final_relative_l2=0.003,
                worst_time_relative_l2=0.005,
                pde_rmse=0.003,
                regional_pde_rmse=0.003,
                worst_time_gradient_relative_l2=0.02,
                grid_refinement_ratio=0.05,
            )
            for name in names
        },
        case_faces={name: CATALOG[name].faces() for name in names},
        problem_sha256={
            name: hashlib.sha256(
                (ROOT / "benchmarks/problems" / name / "functions.py")
                .read_bytes()
                .replace(b"\r\n", b"\n")
            ).hexdigest()
            for name in names
        },
        layer_evaluation=evaluation_identity(),
    )
    m["report_parameters"]["constraints"] = "hard_initial_dirichlet"
    m["physical_limits"] = dict(initial_rmse=1e-6, boundary_max=1e-6)
    return m


def test_unforced_plan_requires_complete_measurement():
    m = unforced_manifest()
    assert plan(m)["trials"] == 2
    assert m["layer_evaluation"]["coarse_grid"] == [801, 201]
    for damage in ("missing", "reference", "gradient", "region", "grid", "boolean", "oldcase"):
        bad = deepcopy(m)
        if damage == "missing":
            del bad["layer_evaluation"]
        elif damage == "reference":
            bad["layer_evaluation"]["reference_sha256"] = "0" * 64
        elif damage == "gradient":
            del bad["case_limits"]["burgers_unforced_nu010"]["worst_time_gradient_relative_l2"]
        elif damage == "region":
            bad["layer_evaluation"]["regional_points"] = 8
        elif damage == "grid":
            bad["layer_evaluation"]["fine_grid"] = [801, 201]
        elif damage == "boolean":
            bad["layer_evaluation"]["regional_time"][1] = True
        else:
            bad["case_limits"]["burgers"] = bad["case_limits"].pop("burgers_unforced_nu010")
        with pytest.raises(ValueError):
            plan(bad)
    assert evaluation_identity()["regional_time"] == [0.15, 1.0]


def test_every_registered_limit_round_trips_through_the_actual_child_parser(monkeypatch, tmp_path):
    import sys

    from benchmarks.cases import CASES
    from benchmarks.convergence import ConvergenceArm, ConvergencePlan
    from benchmarks.convergence_processes import trial_command
    from scripts.benchmark_time_to_quality import arguments

    manifest = unforced_manifest()
    name = "burgers_unforced_nu001"
    expected = ConvergencePlan(
        CASES[name],
        500,
        (0, 100, 250, 500),
        manifest["case_limits"][name],
        "hard_initial_dirichlet",
        1024,
        tolerance_change=1e-12,
    )
    command = trial_command(Path("benchmark.py"), expected, ConvergenceArm(), 11, tmp_path)
    monkeypatch.setattr(sys, "argv", command[1:])
    _, parsed, arms = arguments()
    assert parsed.limits == expected.limits
    assert parsed.steps == 500 and parsed.snapshots == (0, 100, 250, 500)
    assert parsed.points == 1024 and parsed.constraints == "hard_initial_dirichlet"
    assert len(arms) == 1
    without_gradient = command[1:]
    index = without_gradient.index("--worst-time-gradient-relative-l2")
    del without_gradient[index : index + 2]
    monkeypatch.setattr(sys, "argv", without_gradient)
    with pytest.raises(SystemExit) as error:
        arguments()
    assert error.value.code == 2
