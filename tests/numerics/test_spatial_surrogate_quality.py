"""Analytical acceptance must detect a field that ignores the coefficient."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.catalog import CATALOG
from benchmarks.spatial_surrogate_quality import evaluate, evaluation_identity, reference
from pinnstorch.runner.problem import ProblemAdapter, load_problem
from pinnstorch.runner.settings import TaskSettings
from research_tools.planning import plan
from research_tools.surrogate_evidence import validate_snapshots
from tests.numerics.test_absolute_quality import manifest

NAME = "heat_spatial_scale_surrogate"


def spatial_manifest():
    value = manifest()
    source = CASES[NAME].source
    import hashlib

    value.update(
        reports=[f"{NAME}/report.json"],
        expected_trials=[
            dict(case=NAME, seed=seed, **value["absolute_quality"]) for seed in (11, 29, 83)
        ],
        case_limits={
            NAME: dict(
                relative_l2=0.02,
                final_relative_l2=0.03,
                worst_time_relative_l2=0.03,
                pde_rmse=0.02,
                coefficient_response_relative_l2=0.05,
            )
        },
        case_faces={NAME: CATALOG[NAME].faces()},
        problem_sha256={
            NAME: hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        },
        hypothesis_evaluation=evaluation_identity(),
        physical_limits=dict(initial_rmse=1e-6, boundary_max=1e-6),
    )
    return value


class AnalyticalField(torch.nn.Module):
    def __init__(self, ignore_a=False):
        super().__init__()
        self.ignore_a = ignore_a

    def forward(self, features, t):
        x, k = features
        # Keep the graph connected when deliberately ignoring k.
        material = k * 0 + 0.15 if self.ignore_a else k
        return {"u": 4 * x * (1 - x) * torch.exp(-12 * material * t)}


@pytest.mark.parametrize("ignore_a", [False, True])
def test_analytical_acceptance_detects_ignored_coefficient(ignore_a: bool) -> None:
    case = CASES[NAME]
    adapter = ProblemAdapter(
        load_problem(case.source), TaskSettings.read(case.config("cpu", 1, 11)).domain
    )
    metrics, diagnostics = evaluate(AnalyticalField(ignore_a), adapter, "cpu")
    assert [r["a"] for r in diagnostics["hypotheses"]] == evaluation_identity()["hypotheses"]
    assert diagnostics["evaluation_state_unchanged"]
    if ignore_a:
        assert metrics["coefficient_response_relative_l2"] == pytest.approx(1)
        assert metrics["relative_l2"] > 0.1
        assert metrics["pde_rmse"] > 0.1
    else:
        assert max(metrics.values()) < 2e-6
        assert diagnostics["hypotheses"][0]["ignored_a_relative_l2"] > 0.1
        x = np.linspace(0, 1, 7)[:, None]
        t = np.linspace(0, 1, 9)[None, :]
        np.testing.assert_allclose(
            case.reference({"x": x, "a": np.full_like(x, 0.2)}, t),
            reference(x, t, 0.2),
            atol=0,
            rtol=0,
        )


def test_prospective_plan_requires_frozen_reference_and_per_hypothesis_gates() -> None:
    value = spatial_manifest()
    assert plan(value)["trials"] == 3
    for damage in ("missing", "reference", "hypotheses", "response"):
        bad = deepcopy(value)
        if damage == "missing":
            del bad["hypothesis_evaluation"]
        elif damage == "reference":
            bad["hypothesis_evaluation"]["reference_sha256"] = "0" * 64
        elif damage == "hypotheses":
            bad["hypothesis_evaluation"]["hypotheses"].pop()
        else:
            del bad["case_limits"][NAME]["coefficient_response_relative_l2"]
        with pytest.raises(ValueError):
            plan(bad)


def test_compact_evidence_rejects_missing_or_pooled_hypotheses() -> None:
    case = CASES[NAME]
    adapter = ProblemAdapter(
        load_problem(case.source), TaskSettings.read(case.config("cpu", 1, 11)).domain
    )
    metrics, diagnostics = evaluate(AnalyticalField(), adapter, "cpu")
    raw = {"snapshots": [dict(metrics=metrics, **diagnostics)]}
    compact = [
        dict(
            metrics=metrics, boundary_max=max(r["boundary_max"] for r in diagnostics["hypotheses"])
        )
    ]
    value = spatial_manifest()
    validate_snapshots(raw, compact, value)
    for damage in ("missing", "duplicate", "pooled", "state", "identity"):
        bad, summary = deepcopy(raw), deepcopy(compact)
        if damage == "missing":
            bad["snapshots"][0]["hypotheses"].pop()
        elif damage == "duplicate":
            bad["snapshots"][0]["hypotheses"][-1] = deepcopy(bad["snapshots"][0]["hypotheses"][0])
        elif damage == "pooled":
            summary[0]["metrics"]["relative_l2"] = 0.0
        elif damage == "state":
            bad["snapshots"][0]["evaluation_state_unchanged"] = False
        else:
            bad["snapshots"][0]["hypothesis_evaluation"]["ignored_k"] = 0.1
        with pytest.raises(ValueError):
            validate_snapshots(bad, summary, value)


def test_child_parser_keeps_coefficient_response_gate(monkeypatch, tmp_path: Path) -> None:
    import sys

    from benchmarks.convergence import ConvergenceArm, ConvergencePlan
    from benchmarks.convergence_processes import trial_command
    from scripts.benchmark_time_to_quality import arguments

    value = spatial_manifest()
    expected = ConvergencePlan(
        CASES[NAME], 500, (0, 250, 500), value["case_limits"][NAME], "soft", 4096
    )
    command = trial_command(Path("benchmark.py"), expected, ConvergenceArm(), 11, tmp_path)
    monkeypatch.setattr(sys, "argv", command[1:])
    _, parsed, _ = arguments()
    assert parsed.limits == expected.limits
    damaged = command[1:]
    index = damaged.index("--coefficient-response-relative-l2")
    del damaged[index : index + 2]
    monkeypatch.setattr(sys, "argv", damaged)
    with pytest.raises(SystemExit):
        arguments()


def test_shared_case_rejects_pooled_file_benchmark(monkeypatch, tmp_path: Path) -> None:
    import sys

    from scripts.benchmark_solutions import arguments

    output = tmp_path / "pooled"
    monkeypatch.setattr(
        sys, "argv", ["benchmark_solutions.py", "--cases", NAME, "--output-dir", str(output)]
    )
    with pytest.raises(SystemExit) as failure:
        arguments()
    assert failure.value.code == 2
    assert not output.exists()


def test_spatial_protocol_rejects_constant_family_identity() -> None:
    from benchmarks.surrogate_quality import evaluation_identity as constant_identity

    value = spatial_manifest()
    value["hypothesis_evaluation"] = constant_identity()
    with pytest.raises(ValueError, match="per-hypothesis"):
        plan(value)


def test_spatial_evidence_does_not_accept_constant_parameter_rows() -> None:
    case = CASES[NAME]
    adapter = ProblemAdapter(
        load_problem(case.source), TaskSettings.read(case.config("cpu", 1, 11)).domain
    )
    metrics, diagnostics = evaluate(AnalyticalField(), adapter, "cpu")
    diagnostics["hypotheses"][0]["k"] = diagnostics["hypotheses"][0].pop("a")
    with pytest.raises(ValueError, match="hypotheses"):
        validate_snapshots(
            {"snapshots": [dict(metrics=metrics, **diagnostics)]},
            [dict(metrics=metrics, boundary_max=0)],
            spatial_manifest(),
        )
