"""Allowlisted CUDA numerical acceptance in the existing immutable queue."""

from __future__ import annotations

import re
from pathlib import Path

from .schema import Json, digest, number, object_value, read_object

SUITE = "tests/numerics/test_neumann_polynomial_pullback.py"
PREFIX = SUITE + "::"
EXPECTED = [
    *[
        PREFIX + f"test_analytic_pullback_passes_independent_finite_difference_checks[{v}]"
        for v in (False, True)
    ],
    *[
        PREFIX + f"test_analytic_pullback_matches_explicit_physical_derivatives[dtype{i}]"
        for i in (0, 1)
    ],
    *[
        PREFIX
        + f"test_frozen_physical_field_pde_and_parameter_gradient_match_autograd[dtype{i}-{d}]"
        for i in (0, 1)
        for d in (1, 2, 3)
    ],
    PREFIX + "test_analytic_features_remain_no_grad_during_prediction",
    PREFIX + "test_analytic_rule_rejects_nonpolynomial_encoding",
    *[
        PREFIX + f"test_analytic_execution_requires_polynomial_representation[{v}]"
        for v in ("identity", "neumann_cosine")
    ],
    *[
        PREFIX + f"test_shifted_geometry_full_field_identity[dtype{i}-{d}]"
        for i in (0, 1)
        for d in (1, 2, 3)
    ],
]
FIELDS = {
    "contract",
    "kind",
    "registration",
    "revision",
    "suite",
    "tests",
    "test_sha256",
    "environment_requirements",
}
ENVIRONMENT = dict(
    torch="2.1.0+cu118",
    cuda_runtime="11.8",
    default_dtype="torch.float32",
    float32_matmul_precision="highest",
    cuda_matmul_allow_tf32=False,
    cudnn_allow_tf32=False,
)


def validate(manifest: Json) -> Json:
    if (
        set(manifest) != FIELDS
        or manifest["contract"] != 6
        or manifest["kind"] != "numerical-acceptance"
        or manifest["registration"] != "prospective"
        or manifest["suite"] != SUITE
        or manifest["tests"] != EXPECTED
        or manifest["environment_requirements"] != ENVIRONMENT
    ):
        raise ValueError("Expected registered CUDA Neumann acceptance contract 6")
    for key, size in (("revision", 40), ("test_sha256", 64)):
        if (
            not isinstance(manifest[key], str)
            or re.fullmatch(f"[0-9a-f]{{{size}}}", manifest[key]) is None
        ):
            raise ValueError("Invalid acceptance source hash")
    return manifest


def plan(manifest: Json) -> Json:
    validate(manifest)
    return dict(
        contract=1,
        kind="numerical-acceptance-plan",
        revision=manifest["revision"],
        tests=20,
        joint_geometry_tests=6,
        dimensions=[1, 2, 3],
        dtypes=["float32", "float64"],
        optimizer_updates=0,
        training=False,
        performance_measurement=False,
    )


def compare(evidence: Json) -> Json:
    manifest = validate(object_value(evidence["manifest"], "manifest"))
    if evidence.get("contract") != 1 or evidence.get("kind") != "numerical-acceptance-evidence":
        raise ValueError("Expected numerical acceptance evidence")
    if (
        evidence["revision"] != manifest["revision"]
        or evidence["test_sha256"] != manifest["test_sha256"]
        or evidence["source_unchanged"] is not True
        or evidence["environment"] != manifest["environment_requirements"]
        or evidence["environment_after"] != evidence["environment"]
        or evidence["test_device"] != "cuda"
        or evidence["cuda_available"] is not True
        or evidence["gpu"] != "NVIDIA GeForce RTX 4070 Ti"
    ):
        raise ValueError("Acceptance provenance/device differs")
    trials = evidence["trials"]
    if evidence["collected"] != EXPECTED or [t["nodeid"] for t in trials] != EXPECTED:
        raise ValueError("Incomplete acceptance test inventory")
    outcomes = {"passed", "failed", "skipped", "not_run", "xpassed", "xfailed"}
    passed = 0
    for trial in trials:
        if set(trial["phases"]) != {"setup", "call", "teardown"} or any(
            v not in outcomes for v in trial["phases"].values()
        ):
            raise ValueError("Invalid pytest phase report")
        number(trial["seconds"], "test duration")
        passed += all(v == "passed" for v in trial["phases"].values())
    code = evidence["pytest_exit_code"]
    if type(code) is not int or code not in (0, 1):
        raise ValueError("Acceptance interrupted or collection failed")
    return dict(
        contract=1,
        kind="numerical-acceptance-comparison",
        trials=len(trials),
        passed=passed,
        not_passed=len(trials) - passed,
        cuda=True,
        comparison_passed=code == 0 and passed == len(EXPECTED),
        scope="Existing numerical tests only; no training or speed claim",
    )


def collect(root: Path, manifest: Json) -> Json:
    evidence = read_object(root / "acceptance.json")
    if evidence["manifest"] != validate(manifest):
        raise ValueError("Acceptance manifest differs")
    if set(evidence["artifact_sha256"]) != {"pytest.log", "pytest-results.json"}:
        raise ValueError("Incomplete acceptance artifacts")
    for name, checksum in evidence["artifact_sha256"].items():
        if digest(root / name) != checksum:
            raise ValueError("Acceptance artifact checksum differs")
    raw = read_object(root / "pytest-results.json")
    if set(raw) != {"collected", "trials", "pytest_exit_code"} or any(
        raw[k] != evidence[k] for k in raw
    ):
        raise ValueError("Acceptance summary differs from raw reports")
    compare(evidence)
    return evidence
