"""Приведённое время оценивается по исходным физическим кодам и всем нормам."""

import hashlib
from copy import deepcopy

import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.catalog import CATALOG, QUADRATIC_TRAINING_ROWS
from benchmarks.quadratic_surrogate_quality import evaluate, evaluation_identity
from pinnstorch.runner.problem import ProblemAdapter, load_problem
from pinnstorch.runner.settings import TaskSettings
from research_tools.planning import plan
from tests.numerics.test_quadratic_surrogate_quality import AnalyticalField, quadratic_manifest

NAME = "heat_quadratic_clock"


def clock_manifest():
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
    m["network"]["time_scale"] = {"parameter": "a", "reference": 0.15}
    return m


class ClockAnalytical(torch.nn.Module):
    def __init__(self, wrong_clock=False):
        super().__init__()
        self.exact = AnalyticalField()
        self.wrong_clock = wrong_clock

    def forward(self, features, t):
        x, a, b = features
        s = t if self.wrong_clock else a / 0.15 * t
        return self.exact([x, torch.full_like(a, 0.15), b], s)


def test_clock_evaluator_uses_physical_time_and_original_codes():
    case = CASES[NAME]
    adapter = ProblemAdapter(
        load_problem(case.source), TaskSettings.read(case.config("cpu", 1, 11)).domain
    )
    metrics, details = evaluate(ClockAnalytical(), adapter, "cpu", NAME)
    assert max(metrics.values()) < 1e-4, metrics
    identity = evaluation_identity(NAME)
    old = evaluation_identity()
    assert (
        identity["hypotheses"] == old["hypotheses"]
        and identity["response_pairs"] == old["response_pairs"]
    )
    assert identity["training_rows"] != QUADRATIC_TRAINING_ROWS
    assert len(identity["training_rows"]) == 14
    assert identity["splits"] == ["unseen"] * 8 + ["corner"] * 4
    assert details["hypothesis_evaluation"] == identity
    assert details["evaluation_state_unchanged"]
    assert plan(clock_manifest())["trials"] == 3
    wrong, _ = evaluate(ClockAnalytical(True), adapter, "cpu", NAME)
    assert wrong["relative_l2"] > 0.02 and wrong["pde_rmse"] > 0.02


@pytest.mark.parametrize("damage", ["reference", "missing", "training_rows", "analytical_hash"])
def test_clock_quality_protocol_rejects_changed_identity(damage):
    m = deepcopy(clock_manifest())
    if damage == "reference":
        m["network"]["time_scale"]["reference"] = 0.2
    elif damage == "missing":
        del m["network"]["time_scale"]
    elif damage == "training_rows":
        m["hypothesis_evaluation"]["training_rows"] = QUADRATIC_TRAINING_ROWS
    else:
        m["hypothesis_evaluation"]["analytical_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        plan(m)
