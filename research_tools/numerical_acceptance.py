"""Allowlisted CUDA numerical acceptance in the existing immutable queue."""

from __future__ import annotations

import hashlib
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
LIFECYCLE_SUITE = "tests/integration/test_neumann_feature_runner.py"
LIFECYCLE_EXPECTED = [
    LIFECYCLE_SUITE
    + "::test_feature_checkpoint_predict_retrain_and_representation_guard["
    + "-".join(case)
    + "]"
    for case in (
        ("heat_neumann_polynomial1d", "neumann_cosine", "eager"),
        ("heat_neumann_polynomial1d", "neumann_polynomial", "neumann_analytic"),
        ("heat_neumann_polynomial3d", "neumann_polynomial", "eager"),
        ("heat_neumann_polynomial3d", "neumann_polynomial", "neumann_analytic"),
        ("heat_neumann2d", "neumann_polynomial", "eager"),
        ("heat_neumann2d", "neumann_polynomial", "neumann_analytic"),
    )
]
CONDUCTIVITY_SUITE = "tests/numerics/test_variable_conductivity_network.py"
CONDUCTIVITY_EXPECTED = [
    *[
        CONDUCTIVITY_SUITE
        + f"::test_every_network_parameter_matches_expanded_operator[dtype{dtype}-{beta}-{d}]"
        for dtype in (0, 1)
        for beta in (0, 1)
        for d in (1, 2)
    ],
    *[
        CONDUCTIVITY_SUITE
        + f"::test_exact_field_detects_missing_conductivity_derivative[dtype{dtype}-{kind}-{d}]"
        for dtype in (0, 1)
        for kind in ("formula", "table")
        for d in (1, 2)
    ],
]
NONHEAT_SUITE = "tests/numerics/test_nonheat_network.py"
NONHEAT_HELPER = "scripts/audit_nonheat.py"
NONHEAT_EXPECTED = [
    NONHEAT_SUITE + f"::test_nonheat_network_operator_and_every_weight[{seed}-dtype{dtype}-{case}]"
    for seed in (19, 73)
    for dtype in (0, 1)
    for case in ("advection", "burgers")
]
NONHEAT_LIFECYCLE_SUITE = "tests/integration/test_nonheat_runner.py"
NONHEAT_LIFECYCLE_EXPECTED = [
    NONHEAT_LIFECYCLE_SUITE + f"::test_nonheat_train_exact_load_predict_and_retrain[{case}]"
    for case in ("advection", "burgers")
]
BURGERS_HARD_SUITE = "tests/numerics/test_burgers_hard_network.py"
BURGERS_HARD_EXPECTED = [
    BURGERS_HARD_SUITE + f"::test_burgers_hard_operator_and_every_weight[{seed}-dtype{dtype}]"
    for seed in (19, 73)
    for dtype in (0, 1)
]
UNFORCED_SUITE = "tests/numerics/test_unforced_burgers_network.py"
UNFORCED_EXPECTED = [
    UNFORCED_SUITE + f"::test_unforced_operator_and_every_weight[{seed}-dtype{dtype}-{case}]"
    for seed in (11, 29)
    for dtype in (0, 1)
    for case in ("burgers_unforced_nu010", "burgers_unforced_nu001")
]
UNFORCED_LIFECYCLE_SUITE = "tests/integration/test_unforced_burgers_runner.py"
UNFORCED_LIFECYCLE_EXPECTED = [
    UNFORCED_LIFECYCLE_SUITE + f"::test_unforced_public_lifecycle[{case}]"
    for case in ("burgers_unforced_nu010", "burgers_unforced_nu001")
]
CONSTANT_SUITE = "tests/numerics/test_constant_surrogate_network.py"
CONSTANT_EXPECTED = [
    CONSTANT_SUITE + f"::test_shared_constant_operator_and_every_weight[{seed}-dtype{dtype}]"
    for seed in (11, 29)
    for dtype in (0, 1)
]
CONSTANT_LIFECYCLE_SUITE = "tests/integration/test_constant_surrogate_runner.py"
CONSTANT_LIFECYCLE_EXPECTED = [
    CONSTANT_LIFECYCLE_SUITE + "::test_shared_constant_public_lifecycle"
]
SPATIAL_SCALE_SUITE = "tests/numerics/test_spatial_scale_surrogate_network.py"
SPATIAL_SCALE_EXPECTED = [
    *[
        SPATIAL_SCALE_SUITE
        + f"::test_spatial_scale_operator_and_every_weight[{seed}-dtype{dtype}]"
        for seed in (11, 29)
        for dtype in (0, 1)
    ],
    *[
        SPATIAL_SCALE_SUITE
        + f"::test_spatial_scale_analytical_field_detects_missing_coefficient_gradient[dtype{dtype}]"
        for dtype in (0, 1)
    ],
]
SPATIAL_LIFECYCLE_SUITE = "tests/integration/test_spatial_surrogate_runner.py"
SPATIAL_LIFECYCLE_EXPECTED = [SPATIAL_LIFECYCLE_SUITE + "::test_spatial_scale_public_lifecycle"]
FUNCTION_CODE_SUITE = "tests/numerics/test_function_code_network.py"
FUNCTION_CODE_EXPECTED = [
    FUNCTION_CODE_SUITE + f"::test_function_code_operator_and_every_weight[{seed}-dtype{dtype}]"
    for seed in (11, 29)
    for dtype in (0, 1)
]
QUADRATIC_SHAPE_SUITE = "tests/numerics/test_quadratic_shape_network.py"
QUADRATIC_SHAPE_EXPECTED = [
    QUADRATIC_SHAPE_SUITE
    + f"::test_quadratic_shape_operator_and_every_weight[{seed}-dtype{dtype}]"
    for seed in (11, 29)
    for dtype in (0, 1)
]
QUADRATIC_LIFECYCLE_SUITE = "tests/integration/test_quadratic_shape_runner.py"
QUADRATIC_LIFECYCLE_EXPECTED = [
    QUADRATIC_LIFECYCLE_SUITE + "::test_quadratic_shape_public_lifecycle"
]
FUNCTION_CODE_LIFECYCLE_SUITE = "tests/integration/test_function_code_runner.py"
FUNCTION_CODE_LIFECYCLE_EXPECTED = [
    FUNCTION_CODE_LIFECYCLE_SUITE + f"::test_function_code_public_lifecycle[{mode}]"
    for mode in (True, False)
]
QUADRATIC_EVALUATION_SUITE = "tests/integration/test_quadratic_surrogate_convergence.py"
QUADRATIC_EVALUATION_EXPECTED = [
    QUADRATIC_EVALUATION_SUITE + "::test_quadratic_convergence_collects_each_whole_code"
]
HELPER_SUITES = (NONHEAT_SUITE, BURGERS_HARD_SUITE, UNFORCED_SUITE)
SUITES = {
    QUADRATIC_EVALUATION_SUITE: QUADRATIC_EVALUATION_EXPECTED,
    QUADRATIC_LIFECYCLE_SUITE: QUADRATIC_LIFECYCLE_EXPECTED,
    QUADRATIC_SHAPE_SUITE: QUADRATIC_SHAPE_EXPECTED,
    FUNCTION_CODE_SUITE: FUNCTION_CODE_EXPECTED,
    FUNCTION_CODE_LIFECYCLE_SUITE: FUNCTION_CODE_LIFECYCLE_EXPECTED,
    SPATIAL_LIFECYCLE_SUITE: SPATIAL_LIFECYCLE_EXPECTED,
    SPATIAL_SCALE_SUITE: SPATIAL_SCALE_EXPECTED,
    CONSTANT_SUITE: CONSTANT_EXPECTED,
    CONSTANT_LIFECYCLE_SUITE: CONSTANT_LIFECYCLE_EXPECTED,
    UNFORCED_SUITE: UNFORCED_EXPECTED,
    UNFORCED_LIFECYCLE_SUITE: UNFORCED_LIFECYCLE_EXPECTED,
    SUITE: EXPECTED,
    LIFECYCLE_SUITE: LIFECYCLE_EXPECTED,
    CONDUCTIVITY_SUITE: CONDUCTIVITY_EXPECTED,
    NONHEAT_SUITE: NONHEAT_EXPECTED,
    NONHEAT_LIFECYCLE_SUITE: NONHEAT_LIFECYCLE_EXPECTED,
    BURGERS_HARD_SUITE: BURGERS_HARD_EXPECTED,
}
FIELDS = {
    "contract",
    "kind",
    "registration",
    "revision",
    "suite",
    "tests",
    "test_sha256",
    "test_hash_mode",
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
    fields = FIELDS | ({"helper_sha256"} if manifest.get("suite") in HELPER_SUITES else set())
    if (
        set(manifest) != fields
        or manifest["contract"] != 6
        or manifest["kind"] != "numerical-acceptance"
        or manifest["registration"] != "prospective"
        or not isinstance(manifest["suite"], str)
        or manifest["suite"] not in SUITES
        or manifest["tests"] != SUITES[manifest["suite"]]
        or manifest["test_hash_mode"] != "lf-normalized"
        or manifest["environment_requirements"] != ENVIRONMENT
    ):
        raise ValueError("Expected registered allowlisted CUDA acceptance contract 6")
    for key, size in (("revision", 40), ("test_sha256", 64)):
        if (
            not isinstance(manifest[key], str)
            or re.fullmatch(f"[0-9a-f]{{{size}}}", manifest[key]) is None
        ):
            raise ValueError("Invalid acceptance source hash")
    if manifest["suite"] in HELPER_SUITES and (
        not isinstance(manifest["helper_sha256"], str)
        or re.fullmatch("[0-9a-f]{64}", manifest["helper_sha256"]) is None
    ):
        raise ValueError("Invalid numerical helper hash")
    return manifest


def canonical_test_digest(path: Path) -> str:
    """Match Git text across LF/CRLF archives; preserve every other byte."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def plan(manifest: Json) -> Json:
    validate(manifest)
    if manifest["suite"] == QUADRATIC_EVALUATION_SUITE:
        return dict(
            contract=1,
            kind="numerical-acceptance-plan",
            revision=manifest["revision"],
            tests=1,
            training=True,
            performance_measurement=False,
            regular_fits=1,
            adam_steps_per_regular_fit=1,
            lbfgs_outer_steps_per_regular_fit=1,
            hypotheses_per_snapshot=12,
            response_pairs_per_snapshot=6,
            snapshots=2,
        )
    if manifest["suite"] == QUADRATIC_LIFECYCLE_SUITE:
        return dict(
            contract=1,
            kind="numerical-acceptance-plan",
            revision=manifest["revision"],
            tests=1,
            training=True,
            performance_measurement=False,
            regular_fits=1,
            adam_steps_per_regular_fit=3,
            lbfgs_outer_steps_per_regular_fit=2,
            identity_probe_fits=1,
            identity_probe_adam_steps=1,
            identity_probe_learning_rate=1e-300,
            identity_probe_refinement_steps=0,
            prediction_rows=[[0.075, 1.4], [0.125, 1.6]],
            outside_hypothesis_rejected=True,
            changed_decoder_rejected=True,
            changed_order_rejected=True,
        )
    if manifest["suite"] in (FUNCTION_CODE_SUITE, QUADRATIC_SHAPE_SUITE):
        return dict(
            contract=1,
            kind="numerical-acceptance-plan",
            revision=manifest["revision"],
            tests=4,
            case=(
                "heat_function_code"
                if manifest["suite"] == FUNCTION_CODE_SUITE
                else "heat_quadratic_shapes"
            ),
            dtypes=["float32", "float64"],
            seeds=[11, 29],
            network_models=4,
            parameters_per_model=2305,
            finite_difference_checks=4,
            optimizer_updates=0,
            training=False,
            performance_measurement=False,
            parameter_sampling="complete_rows",
        )
    if manifest["suite"] == FUNCTION_CODE_LIFECYCLE_SUITE:
        return dict(
            contract=1,
            kind="numerical-acceptance-plan",
            revision=manifest["revision"],
            tests=2,
            training=True,
            performance_measurement=False,
            regular_fits=2,
            adam_steps_per_regular_fit=3,
            lbfgs_outer_steps_per_regular_fit=2,
            identity_probe_fits=2,
            identity_probe_adam_steps=1,
            identity_probe_learning_rate=1e-300,
            identity_probe_refinement_steps=0,
            prediction_rows=[[0.4, 0.16], [-0.4, 0.16]],
            outside_hypothesis_rejected=True,
            changed_decoder_rejected=True,
            changed_order_rejected=True,
        )
    if manifest["suite"] == SPATIAL_SCALE_SUITE:
        return dict(
            contract=1,
            kind="numerical-acceptance-plan",
            revision=manifest["revision"],
            tests=6,
            cases=["heat_spatial_scale_surrogate"],
            constraints="soft_with_problem_transform",
            dtypes=["float32", "float64"],
            seeds=[11, 29],
            network_models=4,
            parameters_per_model=2273,
            finite_difference_checks=4,
            analytical_field_checks=2,
            optimizer_updates=0,
            training=False,
            performance_measurement=False,
        )
    if manifest["suite"] == CONSTANT_SUITE:
        return dict(
            contract=1,
            kind="numerical-acceptance-plan",
            revision=manifest["revision"],
            tests=4,
            cases=["heat_constant_surrogate"],
            constraints="soft_with_problem_transform",
            dtypes=["float32", "float64"],
            seeds=[11, 29],
            network_models=4,
            parameters_per_model=2273,
            finite_difference_checks=4,
            optimizer_updates=0,
            training=False,
            performance_measurement=False,
        )
    if manifest["suite"] in (CONSTANT_LIFECYCLE_SUITE, SPATIAL_LIFECYCLE_SUITE):
        return dict(
            contract=1,
            kind="numerical-acceptance-plan",
            revision=manifest["revision"],
            tests=1,
            training=True,
            performance_measurement=False,
            regular_fits=1,
            adam_steps_per_regular_fit=3,
            lbfgs_outer_steps_per_regular_fit=2,
            identity_probe_fits=1,
            identity_probe_adam_steps=1,
            identity_probe_learning_rate=1e-300,
            identity_probe_refinement_steps=0,
            prediction_hypotheses=[0.1, 0.2],
            outside_hypothesis_rejected=True,
        )
    if manifest["suite"] in HELPER_SUITES:
        unforced = manifest["suite"] == UNFORCED_SUITE
        hard = manifest["suite"] == BURGERS_HARD_SUITE
        return dict(
            contract=1,
            kind="numerical-acceptance-plan",
            revision=manifest["revision"],
            tests=4 if hard else 8,
            cases=["burgers_unforced_nu010", "burgers_unforced_nu001"]
            if unforced
            else (["burgers"] if hard else ["advection", "burgers"]),
            constraints="hard_initial_dirichlet" if hard or unforced else "soft",
            dtypes=["float32", "float64"],
            seeds=[11, 29] if unforced else [19, 73],
            network_models=4 if hard else 8,
            parameters_per_model=2241,
            finite_difference_checks=2 if hard else 4,
            optimizer_updates=0,
            training=False,
            performance_measurement=False,
        )
    if manifest["suite"] == CONDUCTIVITY_SUITE:
        return dict(
            contract=1,
            kind="numerical-acceptance-plan",
            revision=manifest["revision"],
            tests=16,
            dimensions=[1, 2],
            dtypes=["float32", "float64"],
            coefficient_representations=["formula", "table"],
            beta=[0, 1],
            seeds=[19],
            network_models=16,
            finite_difference_checks=8,
            optimizer_updates=0,
            training=False,
            performance_measurement=False,
        )
    if manifest["suite"] in (LIFECYCLE_SUITE, NONHEAT_LIFECYCLE_SUITE, UNFORCED_LIFECYCLE_SUITE):
        nonheat = manifest["suite"] in (NONHEAT_LIFECYCLE_SUITE, UNFORCED_LIFECYCLE_SUITE)
        return dict(
            contract=1,
            kind="numerical-acceptance-plan",
            revision=manifest["revision"],
            tests=2 if nonheat else 6,
            training=True,
            performance_measurement=False,
            adam_steps_per_regular_fit=3,
            lbfgs_outer_steps_per_regular_fit=2,
            regular_fits=4 if nonheat else 12,
            identity_probe_fits=2 if nonheat else 6,
            identity_probe_adam_steps=1,
            identity_probe_learning_rate=1e-300,
            identity_probe_refinement_steps=0,
            prediction_grid_space_points=5 if nonheat else 4,
            training_grid_space_points=4 if nonheat else 3,
            changed_representation_rejected=not nonheat,
        )
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
    for key in ("test_raw_sha256", "test_raw_sha256_after"):
        if re.fullmatch("[0-9a-f]{64}", evidence[key]) is None:
            raise ValueError("Invalid raw acceptance test hash")
    if evidence["test_raw_sha256"] != evidence["test_raw_sha256_after"]:
        raise ValueError("Raw acceptance source changed")
    if manifest["suite"] in HELPER_SUITES:
        if (
            evidence.get("helper_sha256") != manifest["helper_sha256"]
            or evidence.get("helper_sha256_after") != manifest["helper_sha256"]
            or not isinstance(evidence.get("helper_raw_sha256"), str)
            or re.fullmatch("[0-9a-f]{64}", evidence["helper_raw_sha256"]) is None
            or evidence.get("helper_raw_sha256_after") != evidence["helper_raw_sha256"]
        ):
            raise ValueError("Numerical helper source changed or differs")
    trials = evidence["trials"]
    expected = SUITES[manifest["suite"]]
    if evidence["collected"] != expected or [t["nodeid"] for t in trials] != expected:
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
        comparison_passed=code == 0 and passed == len(expected),
        scope=(
            "Existing file lifecycle tests; small training only, no accuracy/speed claim"
            if manifest["suite"]
            in (
                LIFECYCLE_SUITE,
                NONHEAT_LIFECYCLE_SUITE,
                UNFORCED_LIFECYCLE_SUITE,
                CONSTANT_LIFECYCLE_SUITE,
                SPATIAL_LIFECYCLE_SUITE,
                FUNCTION_CODE_LIFECYCLE_SUITE,
                QUADRATIC_LIFECYCLE_SUITE,
                QUADRATIC_EVALUATION_SUITE,
            )
            else "Existing numerical tests only; no training or speed claim"
        ),
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
