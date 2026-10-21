"""Real shared-weight training survives the normal evidence publication cycle."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.convergence import ConvergenceArm, ConvergencePlan, convergence_trial
from research_tools.comparison import compare
from research_tools.evidence import collect, read_bundle, write_bundle
from tests.numerics.test_spatial_surrogate_quality import NAME, spatial_manifest

pytestmark = pytest.mark.e2e


def test_spatial_scale_convergence_collects_each_hypothesis(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    manifest = spatial_manifest()
    manifest["report_parameters"].update(points_override=7)
    manifest["expected_trials"] = manifest["expected_trials"][:1]
    plan = ConvergencePlan(
        CASES[NAME], 1, (0, 1), manifest["case_limits"][NAME], "soft", 7, "cpu", 1e-12
    )
    trial = convergence_trial(plan, ConvergenceArm(), 11, tmp_path / "trial")
    for row in trial["snapshots"]:
        assert len(row["hypotheses"]) == 9
        assert row["metrics"]["relative_l2"] == max(r["relative_l2"] for r in row["hypotheses"])
        assert row["evaluation_state_unchanged"]
    # Metadata fixture for serialization only; computation above explicitly runs on CPU.
    environment = dict(
        revision=manifest["revision"],
        working_tree_dirty=False,
        **manifest["environment_requirements"],
    )
    report = dict(
        contract=3,
        status="completed",
        case=NAME,
        limits=plan.limits,
        environment=environment,
        **manifest["report_parameters"],
        trials=[trial],
    )
    (tmp_path / NAME).mkdir()
    (tmp_path / NAME / "report.json").write_text(json.dumps(report))
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
