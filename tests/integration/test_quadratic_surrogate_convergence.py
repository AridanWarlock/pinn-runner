"""Real shared-weight training survives the normal evidence publication cycle."""

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.convergence import ConvergenceArm, ConvergencePlan, convergence_trial
from research_tools.comparison import compare
from research_tools.evidence import collect, read_bundle, write_bundle
from tests.numerics.test_quadratic_surrogate_quality import NAME, quadratic_manifest

pytestmark = pytest.mark.e2e


def test_quadratic_convergence_collects_each_whole_code(tmp_path: Path) -> None:
    audit_quadratic_convergence(tmp_path, quadratic_manifest(), NAME)


def audit_quadratic_convergence(tmp_path, manifest, name):
    torch.set_num_threads(1)
    manifest["report_parameters"].update(
        points_override=None if name == "heat_quadratic_modal" else 7
    )
    manifest["expected_trials"] = manifest["expected_trials"][:1]
    device = os.environ.get("PINN_TEST_DEVICE", "cpu")
    assert device in ("cpu", "cuda")
    plan = ConvergencePlan(
        CASES[name],
        1,
        (0, 1),
        manifest["case_limits"][name],
        "soft",
        None if name == "heat_quadratic_modal" else 7,
        device,
        1e-12,
    )
    trial = convergence_trial(plan, ConvergenceArm(), 11, tmp_path / "trial")
    assert (trial["peak_allocated_bytes"] is not None) == (device == "cuda")
    for row in trial["snapshots"]:
        assert len(row["hypotheses"]) == 12
        assert row["metrics"]["relative_l2"] == max(r["relative_l2"] for r in row["hypotheses"])
        assert row["evaluation_state_unchanged"]
        assert len(row["responses"]) == 6
    # Metadata fixture for serialization only; computation above explicitly runs on CPU.
    environment = dict(
        revision=manifest["revision"],
        working_tree_dirty=False,
        **manifest["environment_requirements"],
    )
    report = dict(
        contract=3,
        status="completed",
        case=name,
        limits=plan.limits,
        environment=environment,
        **manifest["report_parameters"],
        trials=[trial],
    )
    (tmp_path / name).mkdir()
    (tmp_path / name / "report.json").write_text(json.dumps(report))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(manifest))
    evidence = collect(tmp_path, manifest, protocol)
    result = compare(evidence)
    assert result["trials"] == 1 and result["passed"] == 0
    write_bundle(tmp_path / "bundle", evidence)
    assert compare(read_bundle(tmp_path / "bundle")) == result
    damaged = deepcopy(evidence)
    damaged["trials"][0]["snapshots"][-1]["hypotheses"].pop()
    with pytest.raises(ValueError, match="hypotheses"):
        compare(damaged)

    return trial
