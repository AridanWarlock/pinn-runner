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
