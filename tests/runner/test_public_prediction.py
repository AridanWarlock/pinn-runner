"""Cold process metadata and gates; numerical CLI control is separate."""

from copy import deepcopy

import pytest

from research_tools.comparison import compare
from research_tools.evidence import read_bundle, write_bundle
from research_tools.planning import plan
from research_tools.public_prediction import GATES, inventory
from research_tools.schema import validate_manifest
from tests.runner.test_frozen_profile import protocol as native_protocol


def protocol():
    return native_protocol() | dict(
        contract=8,
        kind="public-prediction",
        case="heat_neumann_polynomial3d",
        points=152303,
        repeats=3,
        batch_sizes=[128, 2048],
        space_points=17,
        time_points=31,
        comparison=GATES,
    )


def evidence():
    m = protocol()
    return dict(
        contract=1,
        kind="public-prediction-evidence",
        manifest=m,
        revision=m["revision"],
        environment=m["environment_requirements"],
        gpu="NVIDIA GeForce RTX 4070 Ti",
        source_unchanged=True,
        converted_state_equal=True,
        cross_batch_close=True,
        artifact_sha256={},
        trials=[
            dict(
                repeat=r,
                batch=b,
                rule=k,
                seconds=2 if k == "eager" else 2.01,
                exit_code=0,
                field_shape=[4913, 31],
                finite=True,
                field_sha256="e" * 64,
                environment=deepcopy(m["environment_requirements"]),
                gpu="NVIDIA GeForce RTX 4070 Ti",
                peak_allocated_bytes=10000,
                peak_reserved_bytes=20000,
            )
            for r, b, k in inventory()
        ],
    )


def test_public_protocol_plan_and_compact_roundtrip(tmp_path):
    m = protocol()
    assert validate_manifest(m, tmp_path) == m
    assert plan(m)["processes"] == 12
    assert plan(m)["last_batch_sizes"] == [111, 751]
    e = evidence()
    assert compare(e)["comparison_passed"] is True
    write_bundle(tmp_path / "bundle", e)
    assert compare(read_bundle(tmp_path / "bundle"))["trials"] == 12


@pytest.mark.parametrize(
    "change", ["missing", "field", "environment", "shape", "nan", "conversion", "reserved"]
)
def test_incomplete_or_uncontrolled_public_prediction_is_rejected(change):
    e = evidence()
    t = e["trials"][0]
    if change == "missing":
        e["trials"].pop()
    elif change == "field":
        t["field_sha256"] = "f" * 64
    elif change == "environment":
        t["environment"]["cuda_matmul_allow_tf32"] = True
    elif change == "shape":
        t["field_shape"] = [1, 1]
    elif change == "nan":
        t["seconds"] = float("nan")
    elif change == "conversion":
        e["converted_state_equal"] = False
    else:
        t["peak_reserved_bytes"] = 1
    with pytest.raises(ValueError):
        compare(e)


def test_public_regression_is_a_valid_negative_result():
    e = evidence()
    e["trials"][1]["seconds"] = 3
    assert compare(e)["comparison_passed"] is False


def test_public_budget_rejects_float_or_unknown_command():
    for change in (dict(repeats=3.0), dict(extra_command="arbitrary"), dict(points=1024)):
        with pytest.raises(ValueError):
            validate_manifest(protocol() | change, None)
