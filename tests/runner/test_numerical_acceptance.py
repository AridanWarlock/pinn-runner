"""Acceptance transport/phase integrity; these fixtures do not simulate PDE accuracy."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from research_tools import numerical_acceptance as acceptance
from research_tools.comparison import compare
from research_tools.evidence import read_bundle, write_bundle
from research_tools.planning import plan
from research_tools.schema import digest, validate_manifest
from scripts.research_acceptance import PhaseReport


def protocol() -> dict[str, Any]:
    return dict(
        contract=6,
        kind="numerical-acceptance",
        registration="prospective",
        revision="a" * 40,
        suite=acceptance.SUITE,
        tests=acceptance.EXPECTED.copy(),
        test_sha256="b" * 64,
        environment_requirements=acceptance.ENVIRONMENT.copy(),
    )


def evidence() -> dict[str, Any]:
    m = protocol()
    return dict(
        contract=1,
        kind="numerical-acceptance-evidence",
        manifest=m,
        revision=m["revision"],
        test_sha256=m["test_sha256"],
        source_unchanged=True,
        environment=m["environment_requirements"],
        environment_after=m["environment_requirements"].copy(),
        test_device="cuda",
        cuda_available=True,
        gpu="NVIDIA GeForce RTX 4070 Ti",
        collected=acceptance.EXPECTED.copy(),
        pytest_exit_code=0,
        trials=[
            dict(
                nodeid=n,
                phases=dict(setup="passed", call="passed", teardown="passed"),
                seconds=0.1,
            )
            for n in acceptance.EXPECTED
        ],
    )


def test_registered_inventory_plan_bundle_and_compare(tmp_path: Path) -> None:
    e = evidence()
    assert validate_manifest(e["manifest"], tmp_path) == e["manifest"]
    assert plan(e["manifest"])["joint_geometry_tests"] == 6
    assert compare(e)["passed"] == 20
    write_bundle(tmp_path / "bundle", e)
    assert compare(read_bundle(tmp_path / "bundle"))["comparison_passed"]


@pytest.mark.parametrize("status", ["failed", "skipped", "not_run", "xpassed", "xfailed"])
@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
def test_no_failed_skipped_or_expected_failure_counts_as_success(status: str, phase: str) -> None:
    e = evidence()
    e["trials"][0]["phases"][phase] = status
    e["pytest_exit_code"] = 1
    result = compare(e)
    assert result["passed"] == 19 and not result["comparison_passed"]


@pytest.mark.parametrize("mutation", ["inventory", "cpu", "precision", "interrupt", "phase"])
def test_reject_incomplete_or_wrong_environment(mutation: str) -> None:
    e = evidence()
    if mutation == "inventory":
        e["trials"].pop()
    elif mutation == "cpu":
        e["test_device"] = "cpu"
    elif mutation == "precision":
        e["environment_after"]["cuda_matmul_allow_tf32"] = True
    elif mutation == "interrupt":
        e["pytest_exit_code"] = 2
    else:
        e["trials"][0]["phases"]["call"] = "unknown"
    with pytest.raises(ValueError):
        compare(e)


def test_collect_verifies_raw_phase_reports_and_hashes(tmp_path: Path) -> None:
    e = evidence()
    raw = {key: copy.deepcopy(e[key]) for key in ("collected", "trials", "pytest_exit_code")}
    (tmp_path / "pytest-results.json").write_text(json.dumps(raw))
    (tmp_path / "pytest.log").write_text("fixture transport only")
    e["artifact_sha256"] = {n: digest(tmp_path / n) for n in ("pytest-results.json", "pytest.log")}
    (tmp_path / "acceptance.json").write_text(json.dumps(e))
    assert acceptance.collect(tmp_path, e["manifest"]) == e
    raw["trials"][0]["phases"]["call"] = "failed"
    (tmp_path / "pytest-results.json").write_text(json.dumps(raw))
    e["artifact_sha256"]["pytest-results.json"] = digest(tmp_path / "pytest-results.json")
    (tmp_path / "acceptance.json").write_text(json.dumps(e))
    with pytest.raises(ValueError, match="summary differs"):
        acceptance.collect(tmp_path, e["manifest"])


def test_hook_captures_real_pytest_failure_skip_and_teardown(tmp_path: Path) -> None:
    source = tmp_path / "test_probe.py"
    source.write_text("""import pytest
def test_pass(): pass
@pytest.mark.skip(reason="transport probe")
def test_skip(): pass
def test_fail(): assert False
@pytest.fixture
def teardown_error():
    yield
    raise RuntimeError("transport teardown probe")
def test_teardown(teardown_error): pass
""")
    report = PhaseReport()
    code = pytest.main(
        [
            str(source),
            "--rootdir",
            str(tmp_path),
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "--tb=no",
            "-q",
        ],
        plugins=[report],
    )
    assert code == 1
    assert len(report.collected) == 4
    records = {k.split("::")[-1]: v for k, v in report.records.items()}
    assert records["test_pass"]["phases"]["call"] == "passed"
    assert records["test_skip"]["phases"]["setup"] == "skipped"
    assert records["test_fail"]["phases"]["call"] == "failed"
    assert records["test_teardown"]["phases"]["call"] == "passed"
    assert records["test_teardown"]["phases"]["teardown"] == "failed"


def test_protocol_cannot_select_arbitrary_files() -> None:
    m = protocol()
    m["suite"] = "run.py"
    with pytest.raises(ValueError):
        acceptance.validate(m)
