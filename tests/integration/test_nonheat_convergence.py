"""Actual soft-PDE trials feed single-configuration evidence, including failed targets."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.convergence import ConvergenceArm, ConvergencePlan, convergence_trial
from benchmarks.policies import CartesianGrid
from research_tools.comparison import compare
from research_tools.evidence import collect, read_bundle, write_bundle
from tests.numerics.test_absolute_quality import manifest

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("name", ["advection", "burgers"])
def test_nonheat_actual_soft_convergence_collects_negative_quality(
    tmp_path: Path, name: str
) -> None:
    torch.set_num_threads(1)
    m = manifest()
    m["case_limits"] = {name: m["case_limits"][name]}
    m["case_faces"] = {name: m["case_faces"][name]}
    m["problem_sha256"] = {name: m["problem_sha256"][name]}
    m["expected_trials"] = [
        t for t in m["expected_trials"] if t["case"] == name and t["seed"] == 19
    ]
    m["reports"] = ["case/report.json"]
    m["report_parameters"]["points_override"] = 7
    case = replace(CASES[name], grid=CartesianGrid((3, 5), (3, 6)))
    p = ConvergencePlan(case, 1, (0, 1), m["case_limits"][name], "soft", 7, "cpu", 1e-12)
    trial = convergence_trial(p, ConvergenceArm(), 19, tmp_path / "trial")
    assert trial["feature_derivatives"] == trial["control_feature_derivatives"] == "autograd"
    assert trial["problem_source_sha256"] == m["problem_sha256"][name]
    assert trial["prediction_points"] == 30
    for s in trial["snapshots"]:
        assert s["metrics"]["worst_time_relative_l2"] >= s["metrics"]["final_relative_l2"]
        assert len(s["boundary_operators"]) == (1 if name == "advection" else 2)
        if name == "advection":
            assert s["boundary_operators"][0]["side"] == "lower"
    # Transport/serialization environment fixture only: no CUDA claim.
    environment = dict(
        revision=m["revision"], working_tree_dirty=False, **m["environment_requirements"]
    )
    report = dict(
        contract=3,
        status="completed",
        case=name,
        limits=p.limits,
        environment=environment,
        **m["report_parameters"],
        trials=[trial],
    )
    (tmp_path / "case").mkdir()
    (tmp_path / "case/report.json").write_text(json.dumps(report))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(m))
    e = collect(tmp_path, m, protocol)
    result = compare(e)
    assert result["trials"] == 1 and result["passed"] == 0 and not result["comparison_passed"]
    assert result["results"][0]["failed_metrics"]
    write_bundle(tmp_path / "bundle", e)
    assert compare(read_bundle(tmp_path / "bundle")) == result
