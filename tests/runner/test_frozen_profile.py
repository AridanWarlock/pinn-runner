"""Frozen profile provenance and real SQLite reductions; no simulated numerical claims."""

from __future__ import annotations

import copy
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from research_tools import frozen_profile
from research_tools.comparison import compare
from research_tools.evidence import read_bundle, write_bundle
from research_tools.planning import plan
from research_tools.schema import validate_manifest


def protocol() -> dict[str, Any]:
    return dict(
        contract=3,
        kind="frozen-profile",
        registration="prospective",
        revision="a" * 40,
        case="heat_neumann_polynomial2d",
        trial=".gpu-jobs/source/r/trial",
        weights="lb-2000.pt",
        weights_sha256="b" * 64,
        config_sha256="c" * 64,
        points=1024,
        environment_requirements=dict(
            torch="2.1.0+cu118",
            cuda_runtime="11.8",
            default_dtype="torch.float32",
            float32_matmul_precision="highest",
            cuda_matmul_allow_tf32=False,
            cudnn_allow_tf32=False,
        ),
    )


def evidence() -> dict[str, Any]:
    manifest = protocol()
    return dict(
        contract=1,
        kind="frozen-profile-evidence",
        manifest=manifest,
        environment=manifest["environment_requirements"],
        trials=[
            dict(
                rule=rule,
                weights_sha256=manifest["weights_sha256"],
                source_unchanged=True,
                profile_exit_code=0,
                trace={"kernels": {"count": 2}},
                artifact_sha256={},
                workload=dict(
                    weights_unchanged=True,
                    weights_sha256=manifest["weights_sha256"],
                    revision=manifest["revision"],
                    feature_derivatives=rule,
                    timing_includes_external_profiler=True,
                ),
            )
            for rule in ("autograd", "analytic")
        ],
    )


def test_profile_plan_has_no_optimizer_updates(tmp_path: Path) -> None:
    assert validate_manifest(protocol(), tmp_path) == protocol()
    result = plan(protocol())
    assert result["optimizer_updates"] == 0
    assert result["processes"] == 2
    assert result["timing_is_profiled"] is True


@pytest.mark.parametrize(
    "changes",
    [
        dict(points=True),
        dict(points=4097),
        dict(revision="main"),
        dict(trial="../outside"),
        dict(trial="/outside"),
        dict(weights="../weights.pt"),
        dict(unknown=1),
        dict(case="burgers"),
        dict(registration="retrospective_import"),
    ],
)
def test_profile_rejects_unregistered_inputs(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        plan(protocol() | changes)


def test_real_sqlite_trace_counts(tmp_path: Path) -> None:
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER)")
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?)", [(10, 30), (15, 45)]
        )
    result = frozen_profile.trace_summary(path)
    assert result["kernels"] == dict(count=2, summed_nanoseconds=50)
    assert result["copies"]["count"] == 0
    # The sum can exceed a union duration: it is deliberately not called wall time.


def test_profile_bundle_checksums_and_no_speed_claim(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    write_bundle(output, evidence())
    result = compare(read_bundle(output))
    assert result["cuda_trace_captured"] is True
    assert result["performance_claim"] is False
    (output / "final.csv").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        read_bundle(output)


@pytest.mark.parametrize("failure", ["no_kernels", "process_failed", "no_workload"])
def test_failed_diagnostic_is_kept_as_negative_evidence(failure: str) -> None:
    value = evidence()
    trial = value["trials"][1]
    if failure == "no_kernels":
        trial["trace"] = {}
    elif failure == "process_failed":
        trial["profile_exit_code"] = 1
    else:
        trial["workload"] = None
    assert compare(value)["comparison_passed"] is False


@pytest.mark.parametrize("key,value", [("weights_sha256", "d" * 64), ("source_unchanged", False)])
def test_frozen_source_changes_rejected(key: str, value: Any) -> None:
    data = copy.deepcopy(evidence())
    data["trials"][0][key] = value
    with pytest.raises(ValueError, match="source changed"):
        compare(data)


@pytest.mark.parametrize("details", [False, True])
def test_profile_compare_cli_accepts_nonconvergence_output(
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
    assert result["cuda_trace_captured"] is True
    assert result["performance_claim"] is False
