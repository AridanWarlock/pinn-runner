"""Протокол интегрального модального обучения не подменяет физическую оценку."""

import hashlib
from copy import deepcopy

import pytest

from benchmarks.cases import CASES
from benchmarks.catalog import CATALOG
from benchmarks.quadratic_surrogate_quality import evaluation_identity
from research_tools.planning import plan
from tests.numerics.test_quadratic_surrogate_quality import quadratic_manifest

NAME = "heat_quadratic_modal"


def modal_manifest():
    m = quadratic_manifest()
    old = "heat_quadratic_shapes"
    m.update(
        reports=[f"{NAME}/report.json"],
        expected_trials=[dict(t, case=NAME) for t in m["expected_trials"]],
        case_limits={NAME: m["case_limits"][old]},
        case_faces={NAME: CATALOG[NAME].faces()},
        problem_sha256={
            NAME: hashlib.sha256(
                CASES[NAME].source.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
        },
        hypothesis_evaluation=evaluation_identity(NAME),
    )
    m["network"].update(quadratic_modal=True, modal_normalization="mode_gap")
    m["report_parameters"]["points_override"] = None
    return m


def test_modal_integral_protocol_preserves_physical_evaluation_and_three_models():
    m = modal_manifest()
    assert plan(m)["trials"] == 3
    identity = m["hypothesis_evaluation"]
    assert identity["hypotheses"] == evaluation_identity()["hypotheses"]
    assert len(identity["response_pairs"]) == 6 and len(identity["training_rows"]) == 14
    assert identity["training_objective"]["positive_time_teacher"] is False
    assert m["report_parameters"]["points_override"] is None


@pytest.mark.parametrize("damage", ["normalization", "modes", "quadrature", "teacher"])
def test_modal_integral_protocol_rejects_semantic_substitution(damage):
    m = deepcopy(modal_manifest())
    if damage == "normalization":
        m["network"]["modal_normalization"] = "identity"
    elif damage == "teacher":
        m["hypothesis_evaluation"]["training_objective"]["positive_time_teacher"] = True
    else:
        m["hypothesis_evaluation"]["training_objective"][damage] = 0
    with pytest.raises(ValueError):
        plan(m)
