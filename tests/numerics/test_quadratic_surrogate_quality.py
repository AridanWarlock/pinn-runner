"""Реальная аналитическая сеть проверяет оценку и отрицательные контроли."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.catalog import CATALOG
from benchmarks.quadratic_reference import series_coefficients
from benchmarks.quadratic_surrogate_quality import evaluate, evaluation_identity
from pinnstorch.runner.problem import ProblemAdapter, load_problem
from pinnstorch.runner.settings import TaskSettings
from research_tools.planning import plan
from research_tools.surrogate_evidence import validate_snapshots
from tests.numerics.test_spatial_surrogate_quality import spatial_manifest

NAME = "heat_quadratic_shapes"


def quadratic_manifest():
    import hashlib

    value = spatial_manifest()
    limits = dict(
        next(iter(value["case_limits"].values())), dx_relative_l2=0.05, flux_relative_l2=0.05
    )
    value.update(
        reports=[f"{NAME}/report.json"],
        expected_trials=[
            dict(case=NAME, seed=seed, **value["absolute_quality"]) for seed in (11, 29, 83)
        ],
        case_limits={NAME: limits},
        case_faces={NAME: CATALOG[NAME].faces()},
        problem_sha256={
            NAME: hashlib.sha256(
                CASES[NAME].source.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
        },
        hypothesis_evaluation=evaluation_identity(),
    )
    return value


class AnalyticalField(torch.nn.Module):
    def __init__(self, ignore=None):
        super().__init__()
        self.ignore = ignore

    def forward(self, features, t):
        x, a, b = features
        if self.ignore == "a":
            a = a * 0 + 0.1
        if self.ignore == "b":
            b = b * 0 + 1.5
        # Коэффициенты ряда зависят от кода, физические x,t остаются в графе.
        result = x * 0 + t * 0
        for code in torch.unique(b.detach()).cpu().tolist():
            rounded = round(code, 6) if x.dtype == torch.float32 else code
            coefficients = torch.tensor(
                series_coefficients(rounded).values, dtype=x.dtype, device=x.device
            )
            nu = (
                torch.arange(1, 65, dtype=x.dtype, device=x.device) * np.pi / np.log1p(1 / rounded)
            )
            r = 1 + x / b
            phi = r**-0.5 * torch.sin(torch.log1p(x / b) * nu)
            value = (coefficients * phi * torch.exp(-a / b**2 * (nu**2 + 0.25) * t)).sum(
                dim=1, keepdim=True
            )
            result = result + torch.where(b == code, value, torch.zeros_like(value))
        return {"u": result}


def adapter():
    case = CASES[NAME]
    return ProblemAdapter(
        load_problem(case.source), TaskSettings.read(case.config("cpu", 1, 11)).domain
    )


@pytest.fixture(scope="module")
def exact_evaluation():
    return evaluate(AnalyticalField(), adapter(), "cpu")


def test_analytical_evaluator_and_registered_plan(exact_evaluation):
    metrics, diagnostics = exact_evaluation
    assert max(metrics.values()) < 1e-4, metrics
    assert len(diagnostics["hypotheses"]) == 12
    assert len(diagnostics["responses"]) == 6
    assert diagnostics["evaluation_state_unchanged"]
    assert plan(quadratic_manifest())["trials"] == 3
    expected = evaluation_identity()
    assert [[r["a"], r["b"]] for r in diagnostics["hypotheses"]] == expected["hypotheses"]
    assert [r["split"] for r in diagnostics["hypotheses"]] == expected["splits"]


@pytest.mark.parametrize("ignore", ["a", "b"])
def test_evaluator_rejects_ignored_code_component(ignore):
    metrics, _ = evaluate(AnalyticalField(ignore), adapter(), "cpu")
    assert metrics["coefficient_response_relative_l2"] == pytest.approx(1, abs=1e-6)
    assert metrics["relative_l2"] > 0.02


@pytest.mark.parametrize(
    "damage",
    ["missing", "duplicate", "order", "split", "pair", "response", "pooled", "state", "identity"],
)
def test_whole_code_evidence_rejects_corruption(exact_evaluation, damage):
    metrics, diagnostics = deepcopy(exact_evaluation)
    raw = {"snapshots": [dict(metrics=deepcopy(metrics), **diagnostics)]}
    compact = [
        dict(
            metrics=metrics, boundary_max=max(r["boundary_max"] for r in diagnostics["hypotheses"])
        )
    ]
    value = quadratic_manifest()
    validate_snapshots(raw, deepcopy(compact), value)
    snap = raw["snapshots"][0]
    if damage == "missing":
        snap["hypotheses"].pop()
    elif damage == "duplicate":
        snap["hypotheses"][-1] = deepcopy(snap["hypotheses"][0])
    elif damage == "order":
        snap["hypotheses"].reverse()
    elif damage == "split":
        snap["hypotheses"][0]["split"] = "training_corner"
    elif damage == "pair":
        snap["responses"].pop()
    elif damage == "response":
        snap["responses"][0]["right"] = 2
    elif damage == "pooled":
        compact[0]["metrics"]["dx_relative_l2"] = 0
    elif damage == "state":
        snap["evaluation_state_unchanged"] = False
    else:
        snap["hypothesis_evaluation"]["analytical_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_snapshots(raw, compact, value)


@pytest.mark.parametrize(
    "damage", ["evaluator", "analytical", "rows", "pair", "gradient", "flux", "response"]
)
def test_prospective_plan_rejects_changed_contract(damage):
    value = quadratic_manifest()
    if damage == "evaluator":
        value["hypothesis_evaluation"]["reference_sha256"] = "0" * 64
    elif damage == "analytical":
        value["hypothesis_evaluation"]["analytical_sha256"] = "0" * 64
    elif damage == "rows":
        value["hypothesis_evaluation"]["training_rows"].pop()
    elif damage == "pair":
        value["hypothesis_evaluation"]["response_pairs"].pop()
    else:
        key = {
            "gradient": "dx_relative_l2",
            "flux": "flux_relative_l2",
            "response": "coefficient_response_relative_l2",
        }[damage]
        del value["case_limits"][NAME][key]
    with pytest.raises(ValueError):
        plan(value)


def test_child_parser_retains_all_whole_code_gates(monkeypatch, tmp_path: Path):
    import sys

    from benchmarks.convergence import ConvergenceArm, ConvergencePlan
    from benchmarks.convergence_processes import trial_command
    from scripts.benchmark_time_to_quality import arguments

    value = quadratic_manifest()
    expected = ConvergencePlan(
        CASES[NAME], 500, (0, 250, 500), value["case_limits"][NAME], "soft", 4096
    )
    command = trial_command(Path("benchmark.py"), expected, ConvergenceArm(), 11, tmp_path)
    monkeypatch.setattr(sys, "argv", command[1:])
    _, parsed, _ = arguments()
    assert parsed.limits == expected.limits
