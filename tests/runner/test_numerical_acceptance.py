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
        test_hash_mode="lf-normalized",
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
        test_raw_sha256="c" * 64,
        test_raw_sha256_after="c" * 64,
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


def test_lifecycle_inventory_and_budget_are_distinct_from_pure_numerics() -> None:
    e = evidence()
    m = e["manifest"]
    m["suite"] = acceptance.LIFECYCLE_SUITE
    m["tests"] = acceptance.LIFECYCLE_EXPECTED.copy()
    e["collected"] = acceptance.LIFECYCLE_EXPECTED.copy()
    e["trials"] = e["trials"][:6]
    for trial, nodeid in zip(e["trials"], acceptance.LIFECYCLE_EXPECTED):
        trial["nodeid"] = nodeid
    assert plan(m)["training"] is True and plan(m)["regular_fits"] == 12
    assert plan(m)["identity_probe_fits"] == 6
    assert compare(e)["passed"] == 6 and compare(e)["comparison_passed"]
    e["trials"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        compare(e)


def test_variable_conductivity_inventory_preserves_zero_training_budget() -> None:
    e = evidence()
    m = e["manifest"]
    m["suite"] = acceptance.CONDUCTIVITY_SUITE
    m["tests"] = acceptance.CONDUCTIVITY_EXPECTED.copy()
    e["collected"] = acceptance.CONDUCTIVITY_EXPECTED.copy()
    e["trials"] = e["trials"][:16]
    for trial, nodeid in zip(e["trials"], acceptance.CONDUCTIVITY_EXPECTED):
        trial["nodeid"] = nodeid
    planned = plan(m)
    assert planned["tests"] == 16 and planned["network_models"] == 16
    assert planned["finite_difference_checks"] == 8
    assert planned["optimizer_updates"] == 0 and planned["training"] is False
    assert compare(e)["passed"] == 16 and compare(e)["comparison_passed"]
    m["tests"].pop()
    with pytest.raises(ValueError, match="allowlisted"):
        acceptance.validate(m)


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


def test_hash_normalizes_only_git_line_endings_and_keeps_raw_integrity(tmp_path: Path) -> None:
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"def run():\n    return 1\n")
    crlf.write_bytes(b"def run():\r\n    return 1\r\n")
    assert digest(lf) != digest(crlf)
    assert acceptance.canonical_test_digest(lf) == acceptance.canonical_test_digest(crlf)
    crlf.write_bytes(b"def run():\r\n    return 2\r\n")
    assert acceptance.canonical_test_digest(lf) != acceptance.canonical_test_digest(crlf)
    e = evidence()
    e["test_raw_sha256_after"] = "d" * 64
    with pytest.raises(ValueError, match="Raw acceptance source changed"):
        compare(e)


def test_nonheat_inventory_requires_pinned_helper_and_unchanged_sources() -> None:
    e = evidence()
    m = e["manifest"]
    m.update(suite=acceptance.NONHEAT_SUITE, tests=acceptance.NONHEAT_EXPECTED.copy())
    with pytest.raises(ValueError, match="allowlisted"):
        acceptance.validate(m)
    m["helper_sha256"] = "d" * 64
    e.update(
        helper_sha256=m["helper_sha256"],
        helper_sha256_after=m["helper_sha256"],
        helper_raw_sha256="e" * 64,
        helper_raw_sha256_after="e" * 64,
        collected=m["tests"].copy(),
    )
    e["trials"] = [
        dict(nodeid=n, phases=dict(setup="passed", call="passed", teardown="passed"), seconds=0.1)
        for n in m["tests"]
    ]
    planned = plan(m)
    assert planned["tests"] == 8 and planned["finite_difference_checks"] == 4
    assert planned["optimizer_updates"] == 0 and planned["training"] is False
    assert compare(e)["passed"] == 8 and compare(e)["comparison_passed"]
    for key in ("helper_sha256", "helper_sha256_after", "helper_raw_sha256_after"):
        altered = copy.deepcopy(e)
        altered[key] = "f" * 64
        with pytest.raises(ValueError, match="helper"):
            compare(altered)
    m["tests"].pop()
    with pytest.raises(ValueError, match="allowlisted"):
        acceptance.validate(m)


def test_nonheat_lifecycle_is_two_small_file_scenarios_not_quality_training() -> None:
    e = evidence()
    m = e["manifest"]
    m.update(
        suite=acceptance.NONHEAT_LIFECYCLE_SUITE,
        tests=acceptance.NONHEAT_LIFECYCLE_EXPECTED.copy(),
    )
    e["collected"] = m["tests"].copy()
    e["trials"] = e["trials"][:2]
    for trial, nodeid in zip(e["trials"], m["tests"]):
        trial["nodeid"] = nodeid
    planned = plan(m)
    assert planned["tests"] == 2 and planned["training"] is True
    assert planned["regular_fits"] == 4 and planned["identity_probe_fits"] == 2
    assert planned["prediction_grid_space_points"] == 5
    assert compare(e)["passed"] == 2 and compare(e)["comparison_passed"]
    assert "small training only" in compare(e)["scope"]


def test_exact_burgers_control_has_distinct_inventory_and_two_finite_differences() -> None:
    m = protocol()
    m.update(
        suite=acceptance.BURGERS_HARD_SUITE,
        tests=acceptance.BURGERS_HARD_EXPECTED,
        helper_sha256="d" * 64,
    )
    p = plan(m)
    assert p["tests"] == p["network_models"] == 4
    assert p["finite_difference_checks"] == 2 and p["optimizer_updates"] == 0
    assert p["constraints"] == "hard_initial_dirichlet"
    assert p["cases"] == ["burgers"]


def test_quadratic_shape_inventory_has_no_optimizer_and_rejects_incomplete_evidence() -> None:
    e = evidence()
    m = e["manifest"]
    m["suite"] = acceptance.QUADRATIC_SHAPE_SUITE
    m["tests"] = acceptance.QUADRATIC_SHAPE_EXPECTED.copy()
    e["collected"] = m["tests"].copy()
    e["trials"] = e["trials"][:4]
    for trial, nodeid in zip(e["trials"], m["tests"]):
        trial["nodeid"] = nodeid
    planned = plan(m)
    assert planned["case"] == "heat_quadratic_shapes"
    assert planned["tests"] == 4 and planned["parameters_per_model"] == 2305
    assert planned["optimizer_updates"] == 0 and planned["training"] is False
    assert compare(e)["passed"] == 4 and compare(e)["comparison_passed"]
    e["trials"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        compare(e)


@pytest.mark.parametrize(
    "suite", [acceptance.SCALED_TIME_SUITE, acceptance.SCALED_TIME_LIFECYCLE_SUITE]
)
def test_scaled_time_inventory_and_fixed_budget(suite):
    e = evidence()
    m = e["manifest"]
    m.update(suite=suite, tests=acceptance.SUITES[suite].copy())
    e["collected"] = m["tests"].copy()
    e["trials"] = e["trials"][: len(m["tests"])]
    for trial, nodeid in zip(e["trials"], m["tests"]):
        trial["nodeid"] = nodeid
    planned = plan(m)
    assert planned["performance_measurement"] is False
    if suite == acceptance.SCALED_TIME_SUITE:
        assert planned["parameters_per_model"] == 2273
        assert planned["training"] is False and planned["optimizer_updates"] == 0
    else:
        assert planned["adam_steps_per_regular_fit"] == 3
        assert planned["lbfgs_outer_steps_per_regular_fit"] == 2
        assert planned["changed_reference_rejected"] and planned["changed_normalization_rejected"]
    assert compare(e)["passed"] == len(m["tests"])
    assert compare(e)["comparison_passed"]


def test_clock_quality_evaluation_admission_is_one_step_only():
    m = protocol()
    m.update(
        suite=acceptance.CLOCK_EVALUATION_SUITE, tests=acceptance.CLOCK_EVALUATION_EXPECTED.copy()
    )
    p = plan(m)
    assert (
        p["tests"]
        == p["regular_fits"]
        == p["adam_steps_per_regular_fit"]
        == p["lbfgs_outer_steps_per_regular_fit"]
        == 1
    )
    assert p["hypotheses_per_snapshot"] == 12 and p["response_pairs_per_snapshot"] == 6
    assert p["snapshots"] == 2 and p["performance_measurement"] is False


def test_modal_prototype_admission_has_no_training_or_public_contract():
    m = protocol()
    m.update(
        suite=acceptance.MODAL_PROTOTYPE_SUITE, tests=acceptance.MODAL_PROTOTYPE_EXPECTED.copy()
    )
    p = plan(m)
    assert p["parameters_per_model"] == 3199 and p["correction_modes"] == 32
    assert p["tests"] == 4 and p["optimizer_updates"] == 0
    assert p["training"] is False and p["public_checkpoint_contract"] is False


@pytest.mark.parametrize(
    "suite", [acceptance.MODAL_NETWORK_SUITE, acceptance.MODAL_LIFECYCLE_SUITE]
)
def test_public_modal_registered_inventory_preserves_admission_budget(suite):
    e = evidence()
    m = e["manifest"]
    m.update(suite=suite, tests=acceptance.SUITES[suite].copy())
    e["collected"] = m["tests"].copy()
    e["trials"] = e["trials"][: len(m["tests"])]
    for trial, nodeid in zip(e["trials"], m["tests"]):
        trial["nodeid"] = nodeid
    p = plan(m)
    if suite == acceptance.MODAL_NETWORK_SUITE:
        assert p["parameters_per_model"] == 3199
        assert p["all_weight_pde_gradients"] is True
        assert p["training"] is False and p["optimizer_updates"] == 0
    else:
        assert p["adam_steps_per_regular_fit"] == 3
        assert p["lbfgs_outer_steps_per_regular_fit"] == 2
        assert p["identity_probe_learning_rate"] == 1e-300
        assert p["changed_initial_rejected"] and p["changed_modes_rejected"]
    assert compare(e)["comparison_passed"]
    assert compare(e)["passed"] == len(m["tests"])
    e["trials"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        compare(e)
