"""Real full-grid trials retain layer metrics and reject incomplete provenance."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.convergence import ConvergenceArm, ConvergencePlan, convergence_trial
from research_tools.comparison import compare
from research_tools.evidence import collect, compact_trial, read_bundle, write_bundle
from tests.numerics.test_unforced_burgers_protocol import unforced_manifest

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("name", ["burgers_unforced_nu010", "burgers_unforced_nu001"])
def test_unforced_full_grid_convergence_and_evidence(tmp_path: Path, name: str) -> None:
    torch.set_num_threads(1)
    m = unforced_manifest()
    for field in ("case_limits", "case_faces", "problem_sha256"):
        m[field] = {name: m[field][name]}
    m["expected_trials"] = [row for row in m["expected_trials"] if row["case"] == name]
    m["reports"] = ["case/report.json"]
    m["report_parameters"]["points_override"] = 7
    p = ConvergencePlan(
        CASES[name], 1, (0, 1), m["case_limits"][name], "hard_initial_dirichlet", 7, "cpu", 1e-12
    )
    trial = convergence_trial(p, ConvergenceArm(), 11, tmp_path / "trial")
    assert trial["layer_evaluation"] == m["layer_evaluation"]
    assert trial["problem_source_sha256"] == m["problem_sha256"][name]
    assert trial["prediction_points"] == 801 * 201
    for row in trial["snapshots"]:
        assert m["case_limits"][name].keys() <= row["metrics"].keys()
        assert len(row["temporal"]) == 401
        assert row["metrics"]["worst_time_relative_l2"] >= row["metrics"]["final_relative_l2"]
        assert row["metrics"]["initial_rmse"] < 1e-6
    for field in ("layer_evaluation", "problem_source_sha256"):
        damaged = deepcopy(trial)
        del damaged[field]
        with pytest.raises(ValueError):
            compact_trial(damaged, m)
    environment = dict(
        revision=m["revision"], working_tree_dirty=False, **m["environment_requirements"]
    )
    # Serialization fixture only; the actual computation above is explicitly CPU.
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
    evidence = collect(tmp_path, m, protocol)
    result = compare(evidence)
    assert result["trials"] == 1 and result["passed"] == 0
    assert not result["comparison_passed"]
    write_bundle(tmp_path / "bundle", evidence)
    assert compare(read_bundle(tmp_path / "bundle")) == result
